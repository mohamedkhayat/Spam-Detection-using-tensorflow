package com.mohamedkhayat.spamdetector;

import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Stage;
import java.io.IOException;

public class HelpController {
    private Stage primaryStage;
    @FXML
    private Button backButton;
    @FXML
    private AnchorPane pane;
    @FXML
    private Label descriptionLabel,titleLabel;
    @FXML
    public void initialize(){

    }
    public void setPrimaryStage(Stage primaryStage) {
        this.primaryStage = primaryStage;
    }
    public void goBack(javafx.scene.input.MouseEvent mouseEvent)throws IOException {
        FXMLLoader loader = new FXMLLoader(getClass().getResource("main-window.fxml"));
        Scene mainMenu = new Scene(loader.load());
        AppController mainController = loader.getController();
        mainController.setPrimaryStage(primaryStage);
        primaryStage.setScene(mainMenu);
        primaryStage.show();

    }
}
