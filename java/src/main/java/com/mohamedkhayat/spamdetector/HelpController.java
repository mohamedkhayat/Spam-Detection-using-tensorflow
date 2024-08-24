package com.mohamedkhayat.spamdetector;

import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.BorderPane;
import javafx.scene.media.MediaView;
import javafx.scene.media.MediaPlayer;
import javafx.scene.media.Media;
import javafx.stage.Stage;

import java.io.File;
import java.io.IOException;

public class HelpController {
    private Stage primaryStage;
    @FXML
    private MediaView medview;
    @FXML
    private Button backButton;
    @FXML
    private BorderPane bPane;
    @FXML
    public void initialize(){
        String pathToVideo = getClass().getResource("/com/mohamedkhayat/spamdetector/snake.mp4").toExternalForm();
        Media media = new Media(pathToVideo);
        MediaPlayer mediaPlayer = new MediaPlayer(media);
        medview.setMediaPlayer(mediaPlayer);
        mediaPlayer.setOnReady(mediaPlayer::play);
        mediaPlayer.setCycleCount(MediaPlayer.INDEFINITE);
        mediaPlayer.setOnError(() -> {
            System.err.println("Error occurred: " + mediaPlayer.getError().getMessage());
        });
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
