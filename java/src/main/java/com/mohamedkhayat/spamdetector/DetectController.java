package com.mohamedkhayat.spamdetector;

import com.google.gson.JsonParser;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.stage.Stage;
import okhttp3.*;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.util.Objects;

public class DetectController {
    private Stage primarystage;
    @FXML
    private Label titleLabel;
    @FXML
    private TextArea inputField;
    @FXML
    private Label descriptionLabel;
    @FXML
    private Button detectButton;
    @FXML
    private Button backButton;

    public void SetPrimaryStage(Stage primarystage) {
        this.primarystage = primarystage;
    }
    public void goBack(javafx.scene.input.MouseEvent mouseEvent)throws IOException {
        FXMLLoader loader = new FXMLLoader(getClass().getResource("main-window.fxml"));
        Scene mainMenu = new Scene(loader.load());
        AppController mainController = loader.getController();
        mainController.setPrimaryStage(primarystage);
        primarystage.setScene(mainMenu);
        primarystage.show();

    }
    public void Detect(){
        String sentence = inputField.getText();
        if(!sentence.isEmpty()){
            OkHttpClient client = new OkHttpClient();
            Gson gson = new Gson();
            JsonObject jsonObject = new JsonObject();
            jsonObject.addProperty("sentence", sentence);
            String json = gson.toJson(jsonObject);
            RequestBody body = RequestBody.create(json, MediaType.parse("application/json"));
            Request request = new Request.Builder()
                    .url("http:/127.0.0.1:5000/predict")
                    .post(body)
                    .build();
            try (Response response = client.newCall(request).execute()){
                if(response.isSuccessful()){
                    assert response.body() != null;
                    String responseBody = response.body().string();
                    JsonObject responseJson = JsonParser.parseString(responseBody).getAsJsonObject();
                    String result = responseJson.get("result").getAsString();
                    if (Objects.equals(result, "ham")){

                        titleLabel.setText("HAM");
                    }
                    else{

                        titleLabel.setText("SPAM");
                    }
                    System.out.println(result);
                }
                else{
                    throw new IOException("Repose not successful:  " + response);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
    }
}
