module com.mohamedkhayat.spamdetector {
    requires javafx.controls;
    requires javafx.fxml;
    requires javafx.media;
    requires java.desktop;
    requires okhttp3;
    requires com.google.gson;
    opens com.mohamedkhayat.spamdetector to javafx.fxml;
    exports com.mohamedkhayat.spamdetector;
}