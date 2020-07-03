# Use neural nets to predict future revenue in a Shipment Profile Report

![cover](https://jinglescode.github.io/assets/img/posts/time-series-00.webp)

This app is the first prototype from BI's new Machine Learning unit, 'MLinBI'. We aim to build a model that is trained once using a large dataset on a central server, then again using a smaller dataset given to us by the client.

For our prototype, the central server is being skipped and so Training Data refers only to what the client uploads. We aim to demonstrate how this second round of training can occur in real time on the client side using the TensorFlowJS framework.

We will be training the model with the Shipment Profile Report (SPR) from Caroz dating Jun 2018 to Aug 2019. Then we will make predictions for Sep-Nov 2019 and compare this to the observed data for that period.

In production, we would train using as much data as possible from all clients, then use the weights derived here in a process called Transfer Learning to train a new model and make predictions.

Client training is most useful when refining the model for edge cases, often combined with transfer learning. Thus, this application is the perfect use case before you consider that it will reduce the load on Wisecloud servers.

[Explore Demo by the original author](https://jinglescode.github.io/time-series-forecasting-tensorflowjs/)

---
# Part 1 - Training Data
There are 4 parts to this app:
1. Client uploads their Shipment Profile Report
2. We group Sum of Revenue by Departure Date
3. Use this grouped dataset to train a neural network in model.js
4. Graph the previous results and predictions

Requirements for training data:
- csv format
- 'Departure Date' column has yyyy-mm-dd hh:mm:ss format
- 'Sum_of_Revenue' column is an integer or float

These are the default settings for the report. Please upload the data below.

# Part 2 - Clean data
We need to shift the data set that each row incorporates the last 20 days as features, and tomorrows' as the target variable. We use the below function to do that and compute the average.

```javascript
function ComputeSMA(data, window_size)
{
  let r_avgs = [], avg_prev = 0;
  for (let i = 0; i <= data.length - window_size; i++){
    let curr_avg = 0.00, t = i + window_size;
    for (let k = i; k < t && k <= data.length; k++){
      curr_avg += data[k]['price'] / window_size;
    }
    r_avgs.push({ set: data.slice(i, i + window_size), avg: curr_avg });
    avg_prev = curr_avg;
  }
  return r_avgs;
}
```
![Simple Moving Average of Microsoft Corporation closing prices data](https://jinglescode.github.io/assets/img/posts/time-series-02.webp)

# Part 3 - Prediction
Use the TensorFlow.JS framework to make predictions on this dataset.

# Future additions
Prediction Loop:
- This model can only make a prediction one day at a time
- You need to be able to feed tomorrow's prediction back into the training data to make a prediction for the next day
Better Model:
- the model used by the original author sucks ([full code on Github](https://github.com/jinglescode/demos/tree/master/src/app/components/tfjs-timeseries-stocks)).

```
async function trainModel(inputs, outputs, trainingsize, window_size, n_epochs, learning_rate, n_layers, callback){

  const input_layer_shape  = window_size;
  const input_layer_neurons = 100;

  const rnn_input_layer_features = 10;
  const rnn_input_layer_timesteps = input_layer_neurons / rnn_input_layer_features;

  const rnn_input_shape  = [rnn_input_layer_features, rnn_input_layer_timesteps];
  const rnn_output_neurons = 20;

  const rnn_batch_size = window_size;

  const output_layer_shape = rnn_output_neurons;
  const output_layer_neurons = 1;

  const model = tf.sequential();

  let X = inputs.slice(0, Math.floor(trainingsize / 100 * inputs.length));
  let Y = outputs.slice(0, Math.floor(trainingsize / 100 * outputs.length));

  const xs = tf.tensor2d(X, [X.length, X[0].length]).div(tf.scalar(10));
  const ys = tf.tensor2d(Y, [Y.length, 1]).reshape([Y.length, 1]).div(tf.scalar(10));

  model.add(tf.layers.dense({units: input_layer_neurons, inputShape: [input_layer_shape]}));
  model.add(tf.layers.reshape({targetShape: rnn_input_shape}));

  let lstm_cells = [];
  for (let index = 0; index < n_layers; index++) {
       lstm_cells.push(tf.layers.lstmCell({units: rnn_output_neurons}));
  }

  model.add(tf.layers.rnn({
    cell: lstm_cells,
    inputShape: rnn_input_shape,
    returnSequences: false
  }));

  model.add(tf.layers.dense({units: output_layer_neurons, inputShape: [output_layer_shape]}));

  model.compile({
    optimizer: tf.train.adam(learning_rate),
    loss: 'meanSquaredError'
  });

  const hist = await model.fit(xs, ys,
    { batchSize: rnn_batch_size, epochs: n_epochs, callbacks: {
      onEpochEnd: async (epoch, log) => {
        callback(epoch, log);
      }
    }
  });

  return { model: model, stats: hist };
}
```
