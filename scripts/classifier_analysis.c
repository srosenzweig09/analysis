#include <iostream>
#include <ifstream>
using namespace std;

array<float> scores;

// Create a text string, which is used to output the text file
string myText;

// Read from the text file
ifstream MyReadFile("reco_classifier_scores.txt");

counter = 0;
// Use a while loop together with the getline() function to read the file line by line
while (getline (MyReadFile, myText)) {
  // Output the text from the file
  scores.push_back(stof(myText));
  counter++;
}

// Close the file
MyReadFile.close();