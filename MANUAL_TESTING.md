## Manual Testing 

| Scenario                   | Input                                                                                    | Expected Result                                                          | Results |
|----------------------------|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|---------|
| Missing dataset folder     | Temporarily renamed the `data\raw` folder to `data\images` ran program with option 1.    | Program should show a friendly error message.                            |         |
| Invalid image path         | User enters file path `wrong_file.jpg` for option 4.                                     | A readable error is printed rather than the program crashing.            |
| Unsupported file type      | User runs program, attempt to predict and enter a non-image file path, e.g. `README.md`. | OpenCV should fail to read this.                                         |
| Prediction before training | Temporarily deleted the `macro_classifier.joblib` and ran option 4.                      | Should print something like "Train the model before running prediction". |
| Invalid menu choice        | User enters an invalid menu option such as `9` or `a`.                                   | Print an invalid error message and redisplay's the menu.                 |
| Successful end-to-end run  | Run every menu option in order.                                                          | Provides the correct outputs for each menu option.                       |
