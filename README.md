# Time Series Forecasting with TensorFlow.js
> Pull stock prices from online API and perform predictions using Recurrent Neural Network & Long Short Term Memory (LSTM) with TensorFlow.js framework

![cover](https://jinglescode.github.io/assets/img/posts/time-series-00.webp)

Machine learning is becoming increasingly popular these days and a growing number of the world’s population see it is as a magic crystal ball: predicting when and what will happen in the future. This experiment uses artificial neural networks to reveal stock market trends and demonstrates the ability of time series forecasting to predict future stock prices based on past historical data.

Disclaimer: As stock markets fluctuation are dynamic and unpredictable owing to multiple factors, this experiment is 100% educational and by no means a trading prediction tool.

[Explore Demo](https://jinglescode.github.io/demos/tfjs-timeseries-stocks/)

---

# Project Walkthrough
There are 4 parts to this project walkthrough:
1. Get stocks data from online API
2. Compute simple moving average for a given time window
3. Train LSTM neural network
4. Predict and compare predicted values to the actual values

# Get Stocks Data

Before we can train the neural network and make any predictions, we will first require data. The type of data we are looking for is time series: a sequence of numbers in chronological order. A good place to fetch these data are from [alphavantage.co](https://www.alphavantage.co/). This API allows us to retrieve chronological data on specific company stocks prices from the last 20 years.

The API yields the following fields:
- open price
- the highest price of that day
- the lowest price of that day
- closing price (this is used in this project)
- volume

To prepare training dataset for our neural network, we will be using closing stocks price. This also means that we will be aiming to predict the future closing price. Below graph shows 20 years of Microsoft Corporation weekly closing prices.

![20 years of Microsoft Corporation weekly closing prices data from alphavantage.co](https://jinglescode.github.io/assets/img/posts/time-series-01.webp)

# Simple Moving Average

For this experiment, we are using [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning), which means feeding data to the neural network and it learns by mapping input data to the output label. One way to prepare the training dataset is to extract the moving average from that time-series data.

[Simple Moving Average (SMA)](https://www.investopedia.com/terms/s/sma.asp) is a method to identify trends direction for a certain period of time, by looking at the average of all the values within that time window. The number of prices in a time window is selected experimentally.

For example, let’s assume the closing prices for the past 5 days were 13, 15, 14, 16, 17, the SMA would be (13+15+14+16+17)/5 = 15. So the input for our training dataset is the set of prices within a single time window, and its label is the computed moving average of those prices.

Let’s compute the SMA of Microsoft Corporation weekly closing prices data, with a window size of 50.

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

And this is what we get, weekly stock closing price in blue, and SMA in orange. Because SMA is the moving average of 50 weeks, it is smoother than the weekly price, which can fluctuate.

![Simple Moving Average of Microsoft Corporation closing prices data](https://jinglescode.github.io/assets/img/posts/time-series-02.webp)

# Training Data

We can prepare the training data with weekly stock prices and the computed SMA. Given the window size is 50, this means that we will use the closing price of every 50 consecutive weeks as our training features (X), and the SMA of those 50 weeks as our training label (Y). Which [looks like that](https://gist.github.com/jinglescode/60f8f9357b3960a1b3017d7483f8194c).

Next, we split our data into 2 sets, training and validation set. If 70% of the data is used for training, then 30% for validation. The API returns us approximate 1000 weeks of data, so 700 for training, and 300 for validation.

# Train Neural Network

Now that the training data is ready, it is time to create a model for time series prediction, to achieve this we will use [TensorFlow.js](https://www.tensorflow.org/js) framework. TensorFlow.js is a library for developing and training machine learning models in JavaScript, and we can deploy these machine learning capabilities in a web browser.

[Sequential model](https://js.tensorflow.org/api/latest/#sequential) is selected which simply connects each layer and pass the data from input to the output during the training process. In order for the model to learn time series data which are sequential, [recurrent neural network (RNN)](https://js.tensorflow.org/api/latest/#layers.rnn) layer is created and a number of [LSTM cells](https://js.tensorflow.org/api/latest/#layers.lstmCell) are added to the RNN.

The model will be trained using [Adam](https://js.tensorflow.org/api/latest/#train.adam) ([research paper](https://arxiv.org/abs/1412.6980)), a popular optimisation algorithm for machine learning. [Root mean square error](https://js.tensorflow.org/api/latest/#losses.meanSquaredError) which will determine the difference between predicted values and the actual values, so the model is able to learn by minimising the error during the training process.

Here is a code snippet of the model described above, [full code on Github](https://github.com/jinglescode/demos/tree/master/src/app/components/tfjs-timeseries-stocks).

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

These are the [hyper-parameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) (parameters used in the training process) available for tweaking in the [frontend](https://jinglescode.github.io/demos/tfjs-timeseries-stocks):
- Training Dataset Size (%): the amount of data used for training, and remaining data will be used for validation
- Epochs: number of times the dataset is used to train the model ([learn more](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/))
- Learning Rate: the amount of change in the weights during training in each step ([learn more](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/))
- Hidden LSTM Layers: to increase the model complexity to learn in higher dimensional space ([learn more](https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/))

![Web frontend, showing parameters available for tweaking](https://jinglescode.github.io/assets/img/posts/time-series-03.webp)

Click the Begin Training Model button…

![User interface showing training model progress](https://jinglescode.github.io/assets/img/posts/time-series-04.webp)

The model seems to converge at around 15 epoch.

# Validation

Now that the model is trained, it is time to use it for predicting future values, for our case, it is the moving average. We will use the model.predict function from TFJS.

The data has been split into 2 sets, training and validation set. The training set has been used for training the model, thus will be using the validation set to validate the model. Since the model has not seen the validation dataset, it will be good if the model is able to predict values that are close to the true values.

So let us use the remaining data for prediction which allow us to see how closely our predicted values are compared to the actual values.

![The green line denotes the prediction of the validation data, from web demo](https://jinglescode.github.io/assets/img/posts/time-series-05.webp)

Looks like the model predicted (green line) does a good job plotting closely to the actual price (blue line). This means that the model is able to predict the last 30% of the data which was unseen by the model.

Other algorithms can be applied and uses the [Root Mean Square Error](https://www.statisticshowto.datasciencecentral.com/rmse/) to compare 2 or more models performance.

# Prediction

Finally, the model has been validated and the predicted values map closely to its true values, we shall use it to predict the future. We will apply the same model.predict function and use the last 50 data points as the input, because our window size is 50. Since our training data is increment daily, we will use the past 50 days as input, to predict the 51st day.

![Predict the 51st day](https://jinglescode.github.io/assets/img/posts/time-series-06.webp)

# Conclusion

There are many ways to do time series prediction other than using a simple moving average. Possible future work is to implement this with more data from various sources.

With TensorFlow.js, machine learning on a web browser is possible, and it is actually pretty cool.

[Explore the demo on Github](https://jinglescode.github.io/demos/tfjs-timeseries-stocks), this experiment is 100% educational and by no means a trading prediction tool.
