package com.mohamedkhayat.spamdetector;

import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.stage.Stage;

import java.io.IOException;

public class AppController {
    private Stage primaryStage;
    public void setPrimaryStage(Stage primaryStage) {
        this.primaryStage = primaryStage;
    }
    public void openHelp(javafx.scene.input.MouseEvent event) throws IOException {
        FXMLLoader helpLoader = new FXMLLoader(getClass().getResource("help.fxml"));
        Scene helpScene = new Scene(helpLoader.load());
        HelpController controller = helpLoader.getController();
        controller.setPrimaryStage(primaryStage);
        if (primaryStage != null) {
            primaryStage.setScene(helpScene);
            primaryStage.show();
        } else {
            System.out.println("Primary stage is not set!");
        }
    }
    public void openDetect(javafx.scene.input.MouseEvent event) throws IOException {
        FXMLLoader detectLoader = new FXMLLoader(getClass().getResource("detect.fxml"));
        Scene detectScene = new Scene(detectLoader.load());
        DetectController controller= detectLoader.getController();
        controller.SetPrimaryStage(primaryStage);
        if (primaryStage != null) {
            primaryStage.setScene(detectScene);
            primaryStage.show();
        }
        else{
            System.out.println("Primary stage is not set!");
        }

    }
}