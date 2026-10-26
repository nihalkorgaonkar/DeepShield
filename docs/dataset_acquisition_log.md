# DeepShield — Dataset Acquisition Log

**Document Version:** 1.0.0  
**Acquisition Date:** October 5, 2026  
**Target Root Directory:** C:\DeepShieldData (Local NVMe Storage, strictly outside OneDrive)

---

## 1. ASVspoof 2019 Logical Access (LA)

- **Dataset Name:** ASVspoof 2019: The 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge Database (Logical Access)
- **Official Source / Repository:** University of Edinburgh DataShare
- **Official Handle URL:** https://datashare.ed.ac.uk/handle/10283/3336
- **Direct Download Bitstream URL:** https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip
- **Version / Release:** Release 1.0 (Official 2019 Challenge Release)
- **License / Access Restrictions:** Open Access for academic research under the ASVspoof 2019 Database Agreement / Creative Commons Attribution.
- **Expected Archive Size:** 7,640,952,520 bytes (~7.12 GB compressed)
- **Expected Uncompressed Disk Usage:** ~18.5 GB
- **Official Checksum (MD5 / ETag):** 30c98f11d8b2bc21f2c257bfd78bb5c5
- **Destination Path:** C:\DeepShieldData\ASVspoof2019_LA
- **Expected Structure & Partitions:**
  - LA/ASVspoof2019_LA_train/flac/ (25,380 FLAC audio files: 2,580 bona fide, 22,800 spoof across algorithms A01–A06)
  - LA/ASVspoof2019_LA_dev/flac/ (24,844 FLAC audio files: 2,548 bona fide, 22,296 spoof across algorithms A01–A06)
  - LA/ASVspoof2019_LA_eval/flac/ (71,237 FLAC audio files: 7,355 bona fide, 63,882 spoof across algorithms A07–A19)
  - LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
  - LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt
  - LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
- **Acquisition Status:** Initializing download.
- **Exact Command / Script:**
  python scripts/download_asvspoof2019.py --data_root C:\DeepShieldData

---

## 2. ASVspoof 2021 Deepfake (DF) Evaluation Subset

- **Dataset Name:** ASVspoof 2021 Challenge — Speech Deepfake Database (DF Task Evaluation Data)
- **Official Source / Repository:** Zenodo (CERN) & Official ASVspoof Consortium
- **Official Repository URL:** https://zenodo.org/records/4835108
- **Official Challenge Keys URL:** https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz
- **Version / Release:** Release 1.0 (Official 2021 Evaluation Phase)
- **License / Access Restrictions:** Open Access for academic evaluation under ASVspoof 2021 Terms & Conditions.
- **Expected Archive Size:**
  - ASVspoof2021_DF_eval_part00.tar.gz (8.04 GB, MD5: 4f2cbae07cf3ede2a1dde3b8d2ee55ea)
  - ASVspoof2021_DF_eval_part01.tar.gz (8.03 GB, MD5: 1578c89ab433c2b60b1dce93bdf8fbec)
  - ASVspoof2021_DF_eval_part02.tar.gz (8.02 GB, MD5: 5497a35f0126e94a1d7a7d26db57b4f7)
  - ASVspoof2021_DF_eval_part03.tar.gz (8.06 GB, MD5: 42b7512ba2943e98a32a53c9608cf83c)
  - Total compressed size: ~32.15 GB
  - Full evaluation keys archive: 25,985,925 bytes (~24.78 MB)
- **Expected Uncompressed Disk Usage:** ~65.0 GB
- **Destination Path:** C:\DeepShieldData\ASVspoof2021_DF
- **Expected Partitions & Protocol Distinction:**
  - *Training / Development Split:* **None in 2021 release** (Official 2021 DF protocol requires models to be trained on ASVspoof 2019 LA train/dev and evaluated zero-shot on 2021 DF).
  - *Evaluation Split:* ASVspoof2021_DF_eval/flac/ (611,829 FLAC files across diverse lossy codecs, bitrates, and unseen generative algorithms).
  - *Protocol & Key Files:* 	rial_metadata.txt with columns [speaker_id, file_id, codec, source, source_task, key, attack_id, trim_status].
- **Acquisition Status:** Initializing download.
- **Exact Command / Script:**
  python scripts/download_asvspoof2021.py --data_root C:\DeepShieldData
