/* below is a NodeJS module to implement a very basic and simple General Regression Neural Network
INPUT PARAMETERS:
train_x   : 2d array of n rows(training size) and m columns(features) :: data type --> double
train_y   : 1d array of size n (actual output correspondng to each training input) :: data type --> double
input     : 1d array of size n (input data whose Y needs to be predicted) :: data type --> double
sigma     : the value of sigma in the Radial Basis Function (defaults to √2):: data type --> double
normalize : whether to normalize train_x or not (generally normalization of training samples gives better predictions) :: data type --> boolean
*/

module.exports = function(train_x, train_y, input, sigma, normalize) {
  //function to calculate Radial Basis Function Kernel value
  function rbf(diff, sig) {
    if (sig != undefined && sig != 0) {
      let coeff = (-0.5 * diff * diff) / (sig * sig);
      return Math.exp(coeff);
    } else {
      return Math.exp(-1 * diff * diff);
    }
  } //end of rbf function
  //function to normalize an one dimensional array
  function normalize_1d_array(arr) {
    let ss = 0;
    for (let i = 0; i < arr.length; i++) {
      ss = ss + arr[i] * arr[i];
    }
    return arr.map(x => x / Math.sqrt(ss));
  } // end of normalize_1d_array function
  if (normalize == true) {
    for (let x of train_x) {
      x = normalize_1d_array(x);
    }
  }
  //function to predict output give training input using GRNN
  function predict(train_x, train_y, input) {
    if (train_x.length != train_y.length) {
      throw new Error("Shape mismatch");
    } else {
      let num = 0,
        den = 0,
        sig_ma = Math.sqrt(2);
      if (sigma != undefined) {
        sig_ma = sigma;
      }
      for (let i = 0; i < train_x.length; i++) {
        let sum = 0;
        for (let j = 0; j < train_x[0].length; j++) {
          let rbf_temp = rbf(input[j] - train_x[i][j], sig_ma);
          sum = sum + rbf_temp;
        }
        num = num + train_y[i] * sum;
        den = den + sum;
      }
      return num / den;
    } // end of predict function
  }
  return predict(train_x, train_y, input);
}; // end of module export
