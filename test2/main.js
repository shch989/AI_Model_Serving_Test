const express = require('express');
const axios = require('axios');

const app = express();
const port = 3000;

// TensorFlow Serving URL
const TF_SERVING_URL = 'http://localhost:8501/v1/models/my_model:predict';

// Middleware to parse JSON request bodies
app.use(express.json());

// POST endpoint for predictions
app.post('/predict', async (req, res) => {
    const startTime = performance.now(); 

    try {
        // Extract data from request body
        const inputData = req.body;

        // Send POST request to TensorFlow Serving
        const response = await axios.post(TF_SERVING_URL, inputData, {
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const endTime = performance.now();
        const executionTimeMs = endTime - startTime;
        const executionTimeSec = executionTimeMs / 1000;

        const endMemoryUsage = process.memoryUsage().rss;
        const memoryUsageMb = endMemoryUsage / (1024 * 1024)

        console.log(`Execution time: ${executionTimeSec.toFixed(4)} seconds`);
        console.log(`Memory usage: ${memoryUsageMb.toFixed(2)} MB`);

        // Send TensorFlow Serving response back to client
        res.json(response.data);
    } catch (error) {
        // Handle errors and send appropriate response
        console.error('Error during prediction:', error.response ? error.response.data : error.message);
        res.status(error.response ? error.response.status : 500).json({
            error: error.response ? error.response.data : 'An unexpected error occurred.'
        });
    }
});

// Start server
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});