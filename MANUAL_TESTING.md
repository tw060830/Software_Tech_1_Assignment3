# Manual Testing

| Scenario                  | Input                                                    | Expected Result                                               | Evidence             |
|---------------------------|----------------------------------------------------------|---------------------------------------------------------------|----------------------|
| Invalid image path        | User enters file path `wrong_file.jpg` for option 4.     | A readable error is printed rather than the program crashing. | `error_1.pdf`        |
| Invalid menu choice       | User enters an invalid menu option such as `9`.          | Print an invalid error message and redisplay's the menu.      | `error_2.pdf`        |
| Missing model file        | User attempts to run option 4 before training the model. | A friendly error message is shown.                            | `error_3.pdf`        |
| Successful end-to-end run | Run every menu option in order.                          | Provides the correct outputs for each menu option.            | `successful_run.pdf` |