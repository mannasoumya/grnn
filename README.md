# Simple General Regression Neural Network for NodeJS
### (With an In-Built CSV Parser)
[![npm version](https://badge.fury.io/js/grnn.svg)](https://www.npmjs.com/package/grnn)
## Description
This is a NodeJS module to use <b>GRNN</b> to predict given a training data.

The <b>"npm"</b> package can be found <b><a href="https://www.npmjs.com/package/grnn" target="__blank" rel="noopener noreferrer">here</a></b>.  

Check out the <b> <a href="https://en.wikipedia.org/wiki/General_regression_neural_network" target="__blank" rel="noopener noreferrer">Wikipedia </a> </b>page to find out more about GRNN or the beginner stuffs <b><a href="https://easyneuralnetwork.blogspot.com/2013/07/grnn-generalized-regression-neural.html" target="__blank" rel="noopener noreferrer">here</a></b>.
##### (This script has no external dependencies)
### Input Parameters
#### <i>Required</i>
double <b>train_x</b>    : 2d array of n rows(training size) and m columns(features)<br>
double <b>train_y</b>    : 1d array of size n (actual output correspondng to each training input)<br>
double <b>test_x</b>    : 2d array of n1 rows(testing size) and m columns(features)<br>
double <b>test_y</b>    : 1d array of size n1 (actual output correspondng to each testing input)<br>
double <b>input</b>      : 1d array of size n (input data whose Y needs to be predicted) <br>
double <b>sigma</b>      : the value of sigma in the Radial Basis Function :: Standard Deviation<br>
boolean <b>normalize</b> : whether to normalize train_x or not (generally normalization of training samples gives better predictions)<br>

### Functions
<b>predict(input)</b> - Returns predicted value of given input <br>
<b>mse()</b> - Returns the Mean Squared Error for the given input <br>
### Variables
<b>ypred[]</b> - Array which have the predicted values for test input data <br>
<b>optimal_sigma</b> - Value of Optimal Sigma ( Minimum MSE ) -- Must be used after calling mse() function 

## How to Use ? (Example)
### Step 1 : Clone repository
```shell
> git clone https://github.com/mannasoumya/grnn.git
```
#### Or

### Step 1 : Install Via npm
```shell
> npm install grnn
```
### Step 2: Import module and use as follows 
Or you can use it in your own code <br>
```javascript
const grnn = require("./grnn"); // assuming cloned repo in cwd; otherwise use appropriate path to grnn.js
// const grnn = require("grnn");  -- > if installed via npm
const train_x = [[1, 2], [5, 6], [10, 11]],
  train_y = [3, 7, 12],
  input = [5.5, 6.5],
  sigma = 2.16,
  normalize = true;
const test_x = [[8.8, 9.8], [13, 14]];
const test_y = [10.8, 15];
const gr = new grnn(train_x, train_y, sigma, normalize, test_x, test_y);
const pred = gr.predict(input);
const mse = gr.mse();
console.log("Prediction:  " + pred);
console.log("MSE:  " + mse);
console.log("Optimal Value of Sigma:  " + gr.optimal_sigma);
console.log("Predicted Values against Test:  "+ gr.ypred);
``` 
### If you are using CSV parser from file to load data
Here is a snapshot of a <b>sample data</b>.<br><br>
Note that the <b>Variable to be Predicted</b> must be in the <b><i>"LAST COLUMN"</i></b> of the csv file.<br><br>
<img src="https://raw.githubusercontent.com/mannasoumya/grnn/master/sample_data_snapshot.PNG"></img>
<br>
<br>
#### Here's a sample code to use the built-in CSV parser

```javascript
const grnn = require("./grnn"); // assuming cloned repo in cwd; otherwise use appropriate path to grnn.js
// const grnn = require("grnn");  -- > if installed via npm
let gr = new grnn(); // initialize constructor with no parameters
const path="..../data.csv", // path to csv file
  header=true; // if your data contain headers
const data = gr.parseCSV(path, header);
const { train_x, 
        test_x, 
        train_y, 
        test_y } = gr.split_test_train(data); // attribute names are train_x, test_x, train_y, test_y
gr = new grnn(train_x, train_y, 0.1180, false, test_x, test_y); // initialized with actual parameters
const mse = gr.mse();
console.log("MSE:  " + mse);
console.log("Optimal Value of Sigma:  " + gr.optimal_sigma);
console.log("Predicted Values against Test:  "+gr.ypred.map(el=>el.toPrecision(4)));
```


<h3> Please contribute and raise issues. Pull requests are welcome. </h3>
