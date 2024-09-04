const express = require('express');
const tf = require('@tensorflow/tfjs'); // TensorFlow.js
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json()); // JSON 요청 처리를 위한 미들웨어

// 필요한 열 목록
const requiredColumns = [
    "amp_temp_1",
    "amp_temp_2",
    "amp_temp_3",
    "amp_temp_4",
    "cpu_temp",
    "interval",
    "pos_1",
    "pos_2",
    "pos_3",
    "pos_4",
    "signal_pack",
    "signal_pick",
    "speed_1",
    "speed_2",
    "speed_3",
    "speed_4",
    "torque_1",
    "torque_2",
    "torque_3",
    "torque_4",
    "vacuum",
    "signal_pick_count",
    "signal_pack_count",
];

let model;

// 모델 로드 함수
async function loadModel() {
    try {
        const modelJsonPath = path.resolve(__dirname, './model/model.json');
        const modelJson = JSON.parse(fs.readFileSync(modelJsonPath, 'utf-8'));

        const weightFiles = modelJson.weightsManifest[0].paths.map(file => {
            return path.join(__dirname, './model', file);
        });

        const weightBuffers = weightFiles.map(file => fs.readFileSync(file));

        const modelArtifacts = {
            modelTopology: modelJson.modelTopology,
            weightSpecs: modelJson.weightsManifest[0].weights,
            weightData: Buffer.concat(weightBuffers),
        };

        model = await tf.loadLayersModel(tf.io.fromMemory(modelArtifacts));
        console.log('Model loaded successfully');
    } catch (err) {
        console.error('Error loading model:', err);
    }
}

// 모델 로드 시도
loadModel();

// 기본 라우트
app.get('/', (req, res) => {
    res.send({ message: "Hello World" });
});

// 예측 라우트
app.post('/predict', async (req, res) => {
    const startTime = performance.now(); 

    try {
        const inputData = req.body;

        // 입력 데이터 유효성 검사
        const missingColumns = requiredColumns.filter(col => !(col in inputData));
        if (missingColumns.length > 0) {
            return res.status(400).json({ error: `Missing columns in input data: ${missingColumns}` });
        }

        // 입력 데이터를 Tensor로 변환
        const inputTensor = tf.tensor2d([requiredColumns.map(col => inputData[col])]);

        // 예측 수행
        const prediction = model.predict(inputTensor);

        // 예측 결과 변환
        const predictionResult = await prediction.dataSync();

        const endTime = performance.now();
        const executionTimeMs = endTime - startTime;
        const executionTimeSec = executionTimeMs / 1000;

        const endMemoryUsage = process.memoryUsage().rss;
        const memoryUsageMb = endMemoryUsage / (1024 * 1024)

        console.log(`Execution time: ${executionTimeSec.toFixed(4)} seconds`);
        console.log(`Memory usage: ${memoryUsageMb.toFixed(2)} MB`);

        // 예측 결과 반환
        res.json({ prediction: Array.from(predictionResult) });

    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

// 서버 실행
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});