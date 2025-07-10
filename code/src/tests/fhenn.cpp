#include <fhenn/fhenn.h>
#include <iostream>

int test() {
  FHENN::Matrix m1({
      {1.0, 1.0, 2.0},
      {3.0, 1.0, 2.0},
      {4.0, 1.0, 5.0},
  });
  FHENN::Matrix m2({
      {1.0, 2.0, 3.0},
      {4.0, 5.0, 6.0},
      {7.0, 8.0, 9.0},
  });
  std::cout << "m1:\n";
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << m1[i][j] << "\t";
    }
    std::cout << "\n";
  }
  std::cout << "m2:\n";
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << m2[i][j] << "\t";
      //   m1[i][j] = 123.321;
    }
    std::cout << "\n";
  }
  m2 = m2 * m1;
  std::cout << "m2 * m1:\n";
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << m2[i][j] << "\t";
    }
    std::cout << "\n";
  }
  return 0;
}

int main(int argc, char **argv) {
  test();
  std::cout << argv[0] << std::endl;
}