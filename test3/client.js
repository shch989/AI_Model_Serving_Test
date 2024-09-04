const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');
const path = require('path');

// Proto file path
const PROTO_PATH = path.join(__dirname, 'predict.proto');

// Load the protobuf definition
const packageDefinition = protoLoader.loadSync(PROTO_PATH);
const protoDescriptor = grpc.loadPackageDefinition(packageDefinition);

// Get the TensorFlow Serving gRPC package
const tfServing = protoDescriptor.tensorflow.serving;

// Create a gRPC client
const client = new tfServing.PredictionService('localhost:8500', grpc.credentials.createInsecure());

async function predict() {
    const request = {
        model_spec: {
            name: 'my_model', // Replace with your model name
            signature_name: 'serving_default'
        },
        instances: [
            {
                // Replace with your actual input data
                amp_temp_1: 1.0,
                amp_temp_2: 2.0,
                amp_temp_3: 3.0,
                amp_temp_4: 4.0,
                cpu_temp: 5.0,
                interval: 6.0,
                pos_1: 7.0,
                pos_2: 8.0,
                pos_3: 9.0,
                pos_4: 10.0,
                signal_pack: 1,
                signal_pick: 2,
                speed_1: 3.0,
                speed_2: 4.0,
                speed_3: 5.0,
                speed_4: 6.0,
                torque_1: 7.0,
                torque_2: 8.0,
                torque_3: 9.0,
                torque_4: 10.0,
                vacuum: 1,
                signal_pick_count: 2,
                signal_pack_count: 3
            }
        ]
    };

    return new Promise((resolve, reject) => {
        client.Predict(request, (error, response) => {
            if (error) {
                return reject(error);
            }
            resolve(response);
        });
    });
}

predict()
    .then(response => {
        console.log('Prediction:', response);
    })
    .catch(error => {
        console.error('Error:', error);
    });
