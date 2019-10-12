# Simple General Regression Neural Network for NodeJS
## Description
This is a NodeJS module to use <b>GRNN</b> to predict given a training data.

Check out the <b> <a href="https://en.wikipedia.org/wiki/General_regression_neural_network">Wikipedia </a> </b>page to find out more about GRNN or the beginner stuffs <b><a href="https://easyneuralnetwork.blogspot.com/2013/07/grnn-generalized-regression-neural.html">here</a></b>.

### Input Parameters
double <b>train_x</b>    : 2d array of n rows(training size) and m columns(features)<br>
double <b>train_y</b>    : 1d array of size n (actual output correspondng to each training input)<br>
double <b>test_x</b>    : 2d array of n1 rows(testing size) and m columns(features)<br>
double <b>test_y</b>    : 1d array of size n (actual output correspondng to each testing input)<br>
double <b>input</b>      : 1d array of size n (input data whose Y needs to be predicted) <br>
double <b>sigma</b>      : the value of sigma in the Radial Basis Function (defaults to √2) :: Standard Deviation<br>
boolean <b>normalize</b> : whether to normalize train_x or not (generally normalization of training samples gives better predictions)<br>

## How to Use ? (Just an example)
### Step 1 : Clone repository
git clone https://github.com/mannasoumya/grnn.git
### Step 2: Import module and use as follows 
Or you can use it in your own code <br>
```javascript
const grnn = require("./grnn"); // assuming cloned repo in cwd; otherwise use appropriate path to grnn.js
let train_x = [[1, 2], [5, 6], [10, 11]],
  train_y = [3, 7, 12],
  input = [5.5, 6.5],
  sigma = 2.16,
  normalize = true;
  let test_x = [[8.8, 9.8], [13, 14]];
let test_y = [10.8, 15];
let gr = new grnn(train_x, train_y,sigma,normalize,test_x,test_y);
let pred=gr.predict(input);
let mse=gr.mse();
console.log("Prediction:  "+pred);
console.log("MSE: "+mse);
``` 
<h3> Please contribute and raise issues. Pull requests are welcome. </h3>
