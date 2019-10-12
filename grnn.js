/* below is a NodeJS module to implement a very basic and simple General Regression Neural Network
INPUT PARAMETERS:
train_x   : 2d array of n rows(training size) and m columns(features) :: data type --> double
train_y   : 1d array of size n (actual output correspondng to each training input) :: data type --> double
test_x   : 2d array of n1 rows(testing size) and m columns(features) :: data type --> double
test_y   : 1d array of size n (actual output correspondng to each testing input) :: data type --> double
input     : 1d array of size n (input data whose Y needs to be predicted) :: data type --> double
sigma     : the value of sigma in the Radial Basis Function (defaults to √2):: data type --> double
normalize : whether to normalize train_x or not (generally normalization of training samples gives better predictions) :: data type --> boolean

*/
class GRNN {
  constructor(train_x, train_y, sigma, normalize, test_x, test_y) {
    this.train_x = train_x;
    this.train_y = train_y;
    this.sigma = sigma;
    this.normalize = normalize;
    this.test_x = test_x;
    this.test_y = test_y;
    this.ypred = [];
  }
  rbf(diff, sig) {
    if (sig != undefined && sig != 0) {
      let coeff = (-0.5 * diff * diff) / (sig * sig);
      return Math.exp(coeff);
    } else {
      return Math.exp(-1 * diff * diff);
    }
  }
  //function to normalize an one dimensional array
  normalize_1d_array(arr) {
    let ss = 0;
    for (let i = 0; i < arr.length; i++) {
      ss = ss + arr[i] * arr[i];
    }
    return arr.map(x => x / Math.sqrt(ss));
  }
  //function to check normalization switch
  checkNormal(normalize, tr_ts_x) {
    if (normalize == true) {
      for (let x of tr_ts_x) {
        x = this.normalize_1d_array(x);
      }
    }
  }
  //function to predict output given training input using GRNN
  prediction(train_x, train_y, input) {
    this.checkNormal(this.normalize, this.train_x);
    if (train_x.length != train_y.length) {
      throw new Error("Shape mismatch");
    } else {
      let num = 0,
        den = 0,
        sig_ma = Math.sqrt(2);
      if (this.sigma != undefined) {
        sig_ma = this.sigma;
      }
      for (let i = 0; i < train_x.length; i++) {
        let sum = 0;
        for (let j = 0; j < train_x[0].length; j++) {
          let rbf_temp = this.rbf(input[j] - train_x[i][j], sig_ma);
          sum = sum + rbf_temp;
        }
        num = num + train_y[i] * sum;
        den = den + sum;
      }
      return num / den;
    }
  }
  //wrapper like function over prediction(...) function
  predict(input) {
    return this.prediction(this.train_x, this.train_y, input);
  }
  //function to calculate mean square error on test data
  mse() {
    this.checkNormal(this.normalize, this.test_x);
    let sum = 0;
    for (let x of this.test_x) {
      this.ypred.push(this.predict(x));
    }
    for (let i = 0; i < this.ypred.length; i++) {
      let err = this.test_y[i] - this.ypred[i];
      sum = sum + err * err;
    }
    return sum / this.ypred.length;
  }
}
module.exports = GRNN;
