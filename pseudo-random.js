exports.pseudo_random = function(start, end, how_many) {
  let ran = [];
  // function to generate random integer between m1 and m2 including m1 and m2
  function random(m1, m2) {
    return m1 + Math.floor(Math.random() * (m2 - m1 + 1));
  }
  /* returns an array of Pseudo Random numbers if the number of random numbers required is in between range.. 
    otherwise returns random numbers with repetition*/

  function generateRandom(m1, m2, n) {
    if (n <= m2 - m1 + 1) {
      let initial = random(m1, m2);
      ran.push(initial);
      let counter = 1;
      while (counter < n) {
        let temp = random(m1, m2);
        if (belongsTo(ran, temp) === false) {
          ran.push(temp);
          counter++;
        }
      }
    } else {
      for (let i = 1; i <= n; i++) {
        ran.push(random(m1, m2));
      }
    }
    return ran;
  }
  //function to check whether an element belongs to an array or not
  function belongsTo(arr, a) {
    for (let i = 0; i < arr.length; i++) {
      if (a === arr[i]) {
        return true;
      }
    }
    return false;
  }
  return generateRandom(start, end, how_many);
};
