// Source rasters for the asset pipeline (PLAN.md §4.1). All are PDS3 rasters in simple cylindrical projection.
const LOLA = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data";
const WAC = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/MDR/WAC_HAPKE";

export const RAW_DIR = "data/raw";

/** Global LOLA DEM, 64 px/deg (~474 m/px), int16 metres × 0.5. Detached label. */
export const LDEM = {
  image: { url: `${LOLA}/lola_gdr/cylindrical/img/ldem_64.img`, file: "ldem_64.img" },
  label: { url: `${LOLA}/lola_gdr/cylindrical/img/ldem_64.lbl`, file: "ldem_64.lbl" },
};

/** SLDEM2015 512 px/deg (~59 m/px) tile covering Tycho (60°S–30°S, 315°E–360°E), float32 km. Detached label. */
export const SLDEM_TYCHO = {
  image: { url: `${LOLA}/sldem2015/tiles/float_img/sldem2015_512_60s_30s_315_360_float.img`, file: "sldem2015_512_60s_30s_315_360_float.img" },
  label: { url: `${LOLA}/sldem2015/tiles/float_img/sldem2015_512_60s_30s_315_360_float.lbl`, file: "sldem2015_512_60s_30s_315_360_float.lbl" },
};

/** LROC WAC Hapke-normalized 643 nm reflectance, 76 px/deg (~400 m/px), ±70° latitude, float32. Attached labels. */
export const WAC_643 = ["N0450", "N1350", "N2250", "N3150", "S0450", "S1350", "S2250", "S3150"].map((tile) => ({
  image: { url: `${WAC}/WAC_HAPKE_643NM_E350${tile}.IMG`, file: `WAC_HAPKE_643NM_E350${tile}.IMG` },
}));

export const ALL_FILES = [LDEM, SLDEM_TYCHO, ...WAC_643].flatMap((s) => [s.label, s.image].filter(Boolean));
