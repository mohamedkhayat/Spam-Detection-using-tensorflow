package com.mohamedkhayat.spamdetector;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;
import javafx.scene.image.Image;
import java.io.IOException;
import java.util.Objects;

public class App extends Application {
    @Override
    public void start(Stage stage) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(App.class.getResource("main-window.fxml"));
        Scene scene = new Scene(fxmlLoader.load());
        AppController appController = fxmlLoader.getController();
        appController.setPrimaryStage(stage);
        Image icon = new Image(Objects.requireNonNull(getClass().getResourceAsStream("/com/mohamedkhayat/spamdetector/icon.png")));
        stage.getIcons().add(icon);
        stage.setTitle("Spam Detector");
        stage.setScene(scene);
        stage.setResizable(false);
        stage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}