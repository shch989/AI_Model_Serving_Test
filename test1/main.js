const express = require('express');
const axios = require('axios');

const app = express();
const port = 3000;

const FASTAPI_URL = 'http://localhost:8000/predict';

app.use(express.json());

app.post('/predict', async (req, res) => {
    const startTime = Date.now();

    try {
        const inputData = req.body;

        const response = await axios.post(FASTAPI_URL, inputData, {
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const endTime = Date.now(); 
        const executionTimeMs = endTime - startTime;
        const executionTimeSec = executionTimeMs / 1000;

        const endMemoryUsage = process.memoryUsage().rss;
        const memoryUsageMb = endMemoryUsage / (1024 * 1024);

        console.log(`Execution time: ${executionTimeSec.toFixed(4)} seconds`);
        console.log(`Memory usage: ${memoryUsageMb.toFixed(2)} MB`);

        res.json(response.data);
    } catch (error) {
        console.error('Error during prediction:', error.response ? error.response.data : error.message);
        res.status(error.response ? error.response.status : 500).json({
            error: error.response ? error.response.data : 'An unexpected error occurred.'
        });
    }
});

app.listen(port, () => {
    console.log(`Express server is running on http://localhost:${port}`);
});