# Scoreboard portrait model

The detector locates one `avatar` class. The template bank assigns one of 29
Agent identities. Use both stages together; detector class zero is not an
Agent identity. Input is a full scoreboard screenshot with portraits in one
column. At least four reliable portraits must support an annotation.

ONNX input: static float32 RGB `[1,3,800,800]`, normalized by 255 and
letterboxed. Output: `[1,30,6]` containing xyxy, confidence, and class.
Embedded NMS uses IoU 0.5. Pipeline settings and class order are in
`models/scoreboard/`. Template correlation and margin are not probabilities.

Training used 37 real V1 screenshots and 180 synthetic screenshots. V2 pixels
were excluded from training and synthesis. Identical ten-Agent roster groups
stayed together; V1 roster groups were excluded from the independent test.
Development data selected the model and thresholds before testing.

Independent test: 70 screenshots, 700 portraits, 695 correct names and boxes at
both IoU 0.5 and 0.75, precision 100%, recall 99.29%, and 66 fully correct
screenshots. Five portraits were missed; none were mislabeled. Aggregate
results are in `evaluation.json`.

Real test data covers 24 classes. Clove, Gekko, Miks, Reyna, and Tejo have
templates and synthetic training but no real test evidence. Several other
classes have only 1–4 examples. Results do not establish reliability on other
interfaces, isolated portraits, or arbitrary portrait arrangements.

The checkpoint is best epoch 17 of 27 completed epochs. Training configuration
freezes the first 11 layers when fine-tuning. Review that setting for a new
dataset or starting checkpoint. Private source screenshots, annotations, and
training records are excluded from this repository.

Four-thread CPU latency on one development frame, five warmups and 30 repeats:
median 75.7 ms, P95 79.1 ms, excluding file decoding. Benchmark other devices
and images separately. The model is not INT8 quantized.
