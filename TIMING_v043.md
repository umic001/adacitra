# Eksperimen timing ADaCITra v04.3

Load Model, Load Images, dan Output Path, lalu klik **RUN LATENCY**.
Eksperimen menjalankan 10 pass berurutan atas semua gambar yang dipilih, dengan
3 warm-up sebelum setiap pass. Dataset 325 gambar menghasilkan 3.250 pengukuran;
30 warm-up tidak masuk statistik. Model evaluator tidak dipanggil atau diubah.
Jangan menjalankan pekerjaan berat lain selama eksperimen.

Hasil disimpan di `OUTPUT/timing/TIMESTAMP_ID/`:

- `raw_inference_times.csv`: nomor run, indeks/nama gambar, pipeline_ms,
  preprocess_ms, inference_ms, dan postprocess_ms.
- `latency_summary.csv`: hasil utama dari `result.speed['inference']`, dengan
  count, inference_mean_ms, inference_median_ms, inference_p95_ms, dan
  equivalent_inference_fps per run serta pooled seluruh sampel.
- `pipeline_timing_summary.csv`: hasil pendukung dengan statistik pipeline
  dalam kolom pipeline_mean_ms, pipeline_median_ms, pipeline_p95_ms dan
  equivalent_pipeline_fps. Scope tidak lagi bercampur dalam satu tabel.
- `metadata.json`: model, device, urutan gambar, parameter, versi lingkungan,
  definisi timing, ringkasan inference pooled, dan status complete/failed/running.
  Pesan selesai menampilkan ringkasan inference ini dan lokasi laporan utama.

Eksperimen baru tidak lagi menghasilkan `timing_summary.csv` gabungan.
File eksperimen lama tetap utuh; perubahan ini tidak mengubah cara pengukuran.

Pipeline adalah wall time pemanggilan model termasuk preprocessing dan
postprocessing, dengan sinkronisasi CUDA bila digunakan. Waktu membaca gambar,
menggambar, GUI, dan menulis CSV berada di luar interval tersebut. Inference internal
diambil dari result.speed Ultralytics; ini bukan nama lain untuk pipeline latency.
P95 memakai interpolasi linear NumPy. FPS = 1000 / mean_ms untuk scope terkait.
Pooled menghitung semua sampel gambar, bukan rata-rata sepuluh nilai P95.
Pengulangan menggunakan model yang sama dalam satu proses eksperimen, bukan cold start.

Input disalin saat mulai; tombol lain dinonaktifkan sementara untuk menghindari
pekerjaan aplikasi yang bersamaan. GUI tetap responsif. Gambar yang tidak dapat
dibaca menghentikan eksperimen dengan status failed, dan data parsial disimpan.
Jika aplikasi ditutup paksa, status dapat tetap running: jangan anggap hasil itu lengkap.

RUN INFERENCE biasa tetap satu pass dan menambah `csv/raw_inference_times.csv`
serta mean/median/P95 pipeline ke `csv/summary_report.csv`. File biasa ini diperbarui
pada run berikutnya; gunakan RUN LATENCY untuk arsip eksperimen yang tidak tertimpa.

Validasi implementasi: 10 × 5 gambar YOLO11n (50 sampel), 22 baris ringkasan dihitung
ulang dari CSV, inference biasa 5 gambar, dan hash evaluator tidak berubah.
Belum dijalankan eksperimen penuh 10 × 325 gambar atau uji timing CUDA/Jetson.
