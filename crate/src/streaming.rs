use std::collections::{HashMap, HashSet};

use crate::lod::{TileId, TileStatus, TILE_SEGMENTS};
use crate::manifest::Manifest;
use crate::tiles::TILE_SIZE;

/// Tiles fetched at once; each is two requests (albedo and DEM), so 8 requests in flight (PLAN.md §6).
pub const MAX_IN_FLIGHT_TILES: usize = 4;
/// GPU memory for resident tiles: their albedo and DEM textures and vertices.
pub const GPU_TILE_BUDGET_BYTES: u64 = 256 << 20;
/// Failed fetches retry after 1 s, 2 s, ...; the tile is unavailable after this many failures.
const MAX_ATTEMPTS: u8 = 3;
const RETRY_BASE_MS: f64 = 1000.0;

/// GPU bytes held by one resident tile: RGBA8 albedo, `u16` DEM and the 12-byte vertex offsets of its grid and
/// skirt ring.
pub const fn tile_gpu_bytes() -> u64 {
    let texels = (TILE_SIZE * TILE_SIZE) as u64;
    let vertices = ((TILE_SEGMENTS + 1) * (TILE_SEGMENTS + 1) + 4 * TILE_SEGMENTS) as u64;
    texels * (4 + 2) + vertices * 12
}

/// How a tile fetch ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FetchResult {
    Loaded,
    /// HTTP 404: the tile does not exist.
    NotFound,
    /// The response has the wrong size.
    Corrupt,
    /// Network or server error, worth retrying.
    Failed,
}

/// What the caller must do with a completed fetch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Completion {
    /// Upload the data into GPU slot `slot`, which held `evicted` until now.
    Upload { slot: usize, evicted: Option<TileId> },
    /// Drop the data: the tile failed, or every slot is in use.
    Discard,
    /// The request was cancelled or superseded; drop the data.
    Stale,
}

#[derive(Clone, Copy, Debug)]
enum Entry {
    /// `attempts` counts earlier failures.
    Loading { request: u64, attempts: u8 },
    Resident { slot: usize, last_used: u64 },
    Failed { attempts: u8, retry_at_ms: f64 },
    Unavailable,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub started: u64,
    pub cancelled: u64,
    pub failed: u64,
    pub evicted: u64,
    pub discarded: u64,
}

/// Decides which tiles to fetch, cancel, keep and evict. GPU slots are numbered `0..capacity`; a slot is only
/// reused after its tile is evicted, so no more than `capacity` tiles are ever resident.
pub struct TileCache {
    manifest: Manifest,
    entries: HashMap<TileId, Entry>,
    capacity: usize,
    max_in_flight: usize,
    slots_allocated: usize,
    next_request: u64,
    in_flight: usize,
    resident: usize,
    stats: CacheStats,
}

impl TileCache {
    pub fn new(manifest: Manifest, capacity: usize, max_in_flight: usize) -> Self {
        Self {
            manifest,
            entries: HashMap::new(),
            // The six face roots are never evicted, so there must be room for them and one more.
            capacity: capacity.max(7),
            max_in_flight: max_in_flight.max(1),
            slots_allocated: 0,
            next_request: 0,
            in_flight: 0,
            resident: 0,
            stats: CacheStats::default(),
        }
    }

    pub fn status(&self, tile: TileId) -> TileStatus {
        if !self.manifest.contains(tile) {
            return TileStatus::Unavailable;
        }
        match self.entries.get(&tile) {
            Some(Entry::Resident { .. }) => TileStatus::Ready,
            Some(Entry::Unavailable) => TileStatus::Unavailable,
            _ => TileStatus::Pending,
        }
    }

    pub fn slot(&self, tile: TileId) -> Option<usize> {
        match self.entries.get(&tile) {
            Some(Entry::Resident { slot, .. }) => Some(*slot),
            _ => None,
        }
    }

    /// Marks a resident tile as used in `frame`; tiles used in the current frame are never evicted.
    /// Returns false if the tile is not resident or was already marked in `frame`.
    pub fn touch(&mut self, tile: TileId, frame: u64) -> bool {
        match self.entries.get_mut(&tile) {
            Some(Entry::Resident { last_used, .. }) if *last_used != frame => {
                *last_used = frame;
                true
            }
            _ => false,
        }
    }

    /// Cancels loads that are no longer requested, then starts the most urgent `requests` while fewer than
    /// `max_in_flight` are loading and a slot will be free for each when it arrives. Returns the
    /// `(tile, request)` pairs to start and to cancel.
    pub fn schedule(&mut self, requests: &[TileId], frame: u64, now_ms: f64) -> (Vec<(TileId, u64)>, Vec<(TileId, u64)>) {
        let wanted: HashSet<_> = requests.iter().copied().collect();
        let mut cancel = Vec::new();
        for (&tile, entry) in &mut self.entries {
            if let Entry::Loading { request, attempts } = *entry {
                if !wanted.contains(&tile) {
                    cancel.push((tile, request));
                    *entry = Entry::Failed { attempts, retry_at_ms: now_ms };
                }
            }
        }
        // Cancelled tiles that had not failed before go back to unknown.
        self.entries.retain(|_, entry| !matches!(entry, Entry::Failed { attempts: 0, .. }));
        self.in_flight -= cancel.len();
        self.stats.cancelled += cancel.len() as u64;

        let evictable = self.entries.iter().filter(|(&tile, entry)| self.is_evictable(tile, entry, frame)).count();
        let mut room = (self.capacity + evictable).saturating_sub(self.resident + self.in_flight);
        let mut start = Vec::new();
        for &tile in requests {
            if self.in_flight >= self.max_in_flight || room == 0 {
                break;
            }
            if !self.manifest.contains(tile) {
                continue;
            }
            let attempts = match self.entries.get(&tile) {
                None => 0,
                Some(&Entry::Failed { attempts, retry_at_ms }) if now_ms >= retry_at_ms => attempts,
                Some(_) => continue,
            };
            let request = self.next_request;
            self.next_request += 1;
            self.entries.insert(tile, Entry::Loading { request, attempts });
            self.in_flight += 1;
            room -= 1;
            start.push((tile, request));
        }
        self.stats.started += start.len() as u64;
        (start, cancel)
    }

    /// Records how the fetch for `request` ended.
    pub fn complete(&mut self, tile: TileId, request: u64, result: FetchResult, frame: u64, now_ms: f64) -> Completion {
        let attempts = match self.entries.get(&tile) {
            Some(&Entry::Loading { request: current, attempts }) if current == request => attempts,
            _ => return Completion::Stale,
        };
        self.in_flight -= 1;
        match result {
            FetchResult::Loaded => match self.take_slot(frame) {
                Some((slot, evicted)) => {
                    self.entries.insert(tile, Entry::Resident { slot, last_used: frame });
                    self.resident += 1;
                    Completion::Upload { slot, evicted }
                }
                None => {
                    self.entries.remove(&tile);
                    self.stats.discarded += 1;
                    Completion::Discard
                }
            },
            FetchResult::NotFound | FetchResult::Corrupt => {
                self.entries.insert(tile, Entry::Unavailable);
                Completion::Discard
            }
            FetchResult::Failed => {
                self.stats.failed += 1;
                let attempts = attempts + 1;
                let entry = if attempts >= MAX_ATTEMPTS {
                    Entry::Unavailable
                } else {
                    Entry::Failed { attempts, retry_at_ms: now_ms + RETRY_BASE_MS * f64::from(1u32 << (attempts - 1)) }
                };
                self.entries.insert(tile, entry);
                Completion::Discard
            }
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn resident(&self) -> usize {
        self.resident
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    /// GPU slots created so far; they are kept for reuse, so this is the memory actually allocated.
    #[cfg(test)]
    pub fn slots_allocated(&self) -> usize {
        self.slots_allocated
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Face roots and tiles used in `frame` stay resident.
    fn is_evictable(&self, tile: TileId, entry: &Entry, frame: u64) -> bool {
        matches!(entry, Entry::Resident { last_used, .. } if *last_used < frame) && tile.level > 0
    }

    /// A never-used slot, or else the slot of the least recently used evictable tile.
    fn take_slot(&mut self, frame: u64) -> Option<(usize, Option<TileId>)> {
        if self.slots_allocated < self.capacity {
            self.slots_allocated += 1;
            return Some((self.slots_allocated - 1, None));
        }
        let (tile, slot) = self
            .entries
            .iter()
            .filter(|(&tile, entry)| self.is_evictable(tile, entry, frame))
            .filter_map(|(&tile, entry)| match *entry {
                Entry::Resident { slot, last_used } => Some((last_used, tile, slot)),
                _ => None,
            })
            .min()
            .map(|(_, tile, slot)| (tile, slot))?;
        self.entries.remove(&tile);
        self.resident -= 1;
        self.stats.evicted += 1;
        Some((slot, Some(tile)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Face;

    const ROOTS: &str = r#"["px",0,0,0,0], ["nx",0,0,0,0], ["py",0,0,0,0], ["ny",0,0,0,0], ["pz",0,0,0,0], ["nz",0,0,0,0]"#;

    /// Roots plus every level-1 and level-2 tile of `px`.
    fn manifest() -> Manifest {
        let level = |l: u32| {
            let n = 1u32 << l;
            let tiles: Vec<_> = (0..n * n).map(|i| format!(r#"["px",{},{},0,0]"#, i % n, i / n)).collect();
            format!(r#"{{"level": {l}, "geometricError": 1, "tiles": [{}]}}"#, tiles.join(","))
        };
        let json = format!(
            r#"{{"version": "v2", "tileSize": 257, "dem": {{"scale": 0.5, "offset": -9200}}, "levels": [{{"level": 0, "geometricError": 1, "tiles": [{ROOTS}]}}, {}, {}]}}"#,
            level(1),
            level(2)
        );
        Manifest::parse(&json).unwrap()
    }

    fn px(level: u8, x: u32, y: u32) -> TileId {
        TileId { face: Face::Px, level, x, y }
    }

    fn level2() -> Vec<TileId> {
        (0..16).map(|i| px(2, i % 4, i / 4)).collect()
    }

    /// Loads `tiles` in `frame`, completing every started fetch successfully.
    fn load_all(cache: &mut TileCache, tiles: &[TileId], frame: u64) -> Vec<Completion> {
        let (start, _) = cache.schedule(tiles, frame, 0.0);
        assert!(start.len() <= cache.max_in_flight);
        start.into_iter().map(|(tile, request)| cache.complete(tile, request, FetchResult::Loaded, frame, 0.0)).collect()
    }

    #[test]
    fn limits_match_the_plan() {
        // Two requests per tile; PLAN.md §6 allows 6-8 in flight.
        assert!((6..=8).contains(&(2 * MAX_IN_FLIGHT_TILES)));
        let capacity = GPU_TILE_BUDGET_BYTES / tile_gpu_bytes();
        assert_eq!(tile_gpu_bytes(), 257 * 257 * 6 + (65 * 65 + 256) * 12);
        assert_eq!(u64::from(crate::geometry::grid_vertex_count(TILE_SEGMENTS)) * 12, tile_gpu_bytes() - 257 * 257 * 6);
        assert!((550..=650).contains(&capacity), "{capacity}");
    }

    #[test]
    fn status_follows_manifest_and_loads() {
        let mut cache = TileCache::new(manifest(), 32, 4);
        assert_eq!(cache.status(px(1, 0, 0)), TileStatus::Pending);
        assert_eq!(cache.status(px(3, 0, 0)), TileStatus::Unavailable);
        assert_eq!(cache.status(TileId { face: Face::Nx, level: 1, x: 0, y: 0 }), TileStatus::Unavailable);
        load_all(&mut cache, &[px(1, 0, 0)], 1);
        assert_eq!(cache.status(px(1, 0, 0)), TileStatus::Ready);
        assert_eq!(cache.slot(px(1, 0, 0)), Some(0));
    }

    #[test]
    fn starts_the_most_urgent_requests_up_to_the_in_flight_limit() {
        let mut cache = TileCache::new(manifest(), 32, 4);
        let requests = level2();
        let (start, cancel) = cache.schedule(&requests, 1, 0.0);
        assert!(cancel.is_empty());
        assert_eq!(start.iter().map(|s| s.0).collect::<Vec<_>>(), requests[..4]);
        assert_eq!(cache.in_flight(), 4);
        // Nothing more starts while all four are loading, and already-loading tiles are not restarted.
        assert!(cache.schedule(&requests, 2, 0.0).0.is_empty());
        let (tile, request) = start[0];
        cache.complete(tile, request, FetchResult::Loaded, 2, 0.0);
        assert_eq!(cache.schedule(&requests, 3, 0.0).0, [(requests[4], 4)]);
        assert_eq!(cache.in_flight(), 4);
    }

    #[test]
    fn unrequested_loads_are_cancelled_and_their_late_results_ignored() {
        let mut cache = TileCache::new(manifest(), 32, 4);
        let requests = level2();
        let (start, _) = cache.schedule(&requests[..4], 1, 0.0);
        let (restart, cancel) = cache.schedule(&requests[2..6], 2, 0.0);
        assert_eq!(cancel.iter().map(|c| c.0).collect::<Vec<_>>().len(), 2);
        assert!(cancel.contains(&start[0]) && cancel.contains(&start[1]));
        assert_eq!(restart.iter().map(|s| s.0).collect::<Vec<_>>(), requests[4..6]);
        assert_eq!(cache.in_flight(), 4);
        assert_eq!(cache.stats().cancelled, 2);
        // A cancelled fetch that still completes is dropped, and a later request starts it afresh.
        assert_eq!(cache.complete(start[0].0, start[0].1, FetchResult::Loaded, 2, 0.0), Completion::Stale);
        assert_eq!(cache.status(start[0].0), TileStatus::Pending);
        assert_eq!(cache.in_flight(), 4);
    }

    #[test]
    fn missing_and_corrupt_tiles_become_unavailable() {
        let mut cache = TileCache::new(manifest(), 32, 4);
        let (start, _) = cache.schedule(&[px(1, 0, 0), px(1, 1, 0)], 1, 0.0);
        assert_eq!(cache.complete(start[0].0, start[0].1, FetchResult::NotFound, 1, 0.0), Completion::Discard);
        cache.complete(start[1].0, start[1].1, FetchResult::Corrupt, 1, 0.0);
        assert_eq!(cache.status(px(1, 0, 0)), TileStatus::Unavailable);
        assert_eq!(cache.status(px(1, 1, 0)), TileStatus::Unavailable);
        assert!(cache.schedule(&[px(1, 0, 0), px(1, 1, 0)], 2, 0.0).0.is_empty());
        assert_eq!(cache.in_flight(), 0);
    }

    #[test]
    fn failures_retry_with_backoff_then_give_up() {
        let mut cache = TileCache::new(manifest(), 32, 4);
        let tile = px(1, 0, 0);
        let mut now = 0.0;
        for (attempt, wait_ms) in [(1, 1000.0), (2, 2000.0)] {
            let (start, _) = cache.schedule(&[tile], attempt, now);
            assert_eq!(start.len(), 1, "attempt {attempt}");
            cache.complete(tile, start[0].1, FetchResult::Failed, attempt, now);
            assert_eq!(cache.status(tile), TileStatus::Pending);
            assert!(cache.schedule(&[tile], attempt, now + wait_ms - 1.0).0.is_empty(), "retried too early");
            now += wait_ms;
        }
        let (start, _) = cache.schedule(&[tile], 3, now);
        cache.complete(tile, start[0].1, FetchResult::Failed, 3, now);
        assert_eq!(cache.status(tile), TileStatus::Unavailable);
        assert_eq!(cache.stats().failed, 3);
    }

    #[test]
    fn full_cache_evicts_least_recently_used_but_never_roots_or_tiles_in_use() {
        let roots: Vec<_> = Face::ALL.into_iter().map(TileId::root).collect();
        let mut cache = TileCache::new(manifest(), 10, 16);
        load_all(&mut cache, &roots, 1);
        let level1: Vec<_> = (0..4).map(|i| px(1, i % 2, i / 2)).collect();
        load_all(&mut cache, &level1[..2], 2);
        load_all(&mut cache, &level1[2..], 3);
        assert_eq!((cache.resident(), cache.slots_allocated()), (10, 10));

        // Frame 4 uses the two tiles loaded in frame 3: only the two from frame 2 can make room.
        cache.touch(level1[2], 4);
        cache.touch(level1[3], 4);
        let completions = load_all(&mut cache, &level2()[..3], 4);
        assert_eq!(
            completions,
            [
                Completion::Upload { slot: 6, evicted: Some(level1[0]) },
                Completion::Upload { slot: 7, evicted: Some(level1[1]) },
            ],
            "only two slots could be freed, so only two loads may start"
        );
        assert_eq!(cache.resident(), 10);
        assert!(roots.iter().all(|&r| cache.status(r) == TileStatus::Ready));
    }

    #[test]
    fn data_arriving_after_its_room_vanished_is_discarded() {
        let roots: Vec<_> = Face::ALL.into_iter().map(TileId::root).collect();
        let mut cache = TileCache::new(manifest(), 7, 8);
        load_all(&mut cache, &roots, 1);
        load_all(&mut cache, &[px(1, 0, 0)], 2);
        // Frame 3 starts a load that can evict px(1, 0, 0); frame 4 uses that tile again before the data arrives.
        let (start, _) = cache.schedule(&[px(1, 1, 0)], 3, 0.0);
        assert_eq!(start.len(), 1);
        cache.touch(px(1, 0, 0), 4);
        assert_eq!(cache.complete(start[0].0, start[0].1, FetchResult::Loaded, 4, 0.0), Completion::Discard);
        assert_eq!(cache.status(px(1, 1, 0)), TileStatus::Pending);
        assert_eq!(cache.status(px(1, 0, 0)), TileStatus::Ready);
        assert_eq!((cache.resident(), cache.stats().discarded), (7, 1));
    }

    /// Random requests, completions and touches never push residency or loads over their limits.
    #[test]
    fn limits_hold_under_random_traffic() {
        let mut cache = TileCache::new(manifest(), 12, 4);
        let pool: Vec<_> = Face::ALL.into_iter().map(TileId::root).chain((0..4).map(|i| px(1, i % 2, i / 2))).chain(level2()).collect();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rand = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        let mut loading: Vec<(TileId, u64)> = Vec::new();
        for frame in 1..2000u64 {
            let requests: Vec<_> = (0..rand(8)).map(|_| pool[rand(pool.len())]).collect();
            for _ in 0..rand(6) {
                cache.touch(pool[rand(pool.len())], frame);
            }
            let (start, cancel) = cache.schedule(&requests, frame, frame as f64 * 16.0);
            loading.retain(|l| !cancel.contains(l));
            loading.extend(start);
            while !loading.is_empty() && rand(3) == 0 {
                let (tile, request) = loading.swap_remove(rand(loading.len()));
                let result = [FetchResult::Loaded, FetchResult::Loaded, FetchResult::Loaded, FetchResult::Failed][rand(4)];
                cache.complete(tile, request, result, frame, frame as f64 * 16.0);
            }
            assert!(cache.in_flight() <= 4 && cache.in_flight() == loading.len(), "frame {frame}");
            assert!(cache.resident() <= cache.capacity() && cache.slots_allocated() <= cache.capacity());
            let slots: HashSet<_> = pool.iter().filter_map(|&t| cache.slot(t)).collect();
            assert_eq!(slots.len(), cache.resident(), "frame {frame}: two tiles share a slot");
        }
        assert!(cache.stats().evicted > 0 && cache.stats().cancelled > 0);
    }
}
