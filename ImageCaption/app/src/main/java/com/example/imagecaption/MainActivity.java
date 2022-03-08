package com.example.imagecaption;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import android.Manifest;
import android.annotation.SuppressLint;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Handler;
import android.provider.MediaStore;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Locale;
import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;


public class MainActivity extends AppCompatActivity {

    public static final  int CAMERA_ACTION_CODE = 1;
    public static final  int UPLOAD_ACTION_CODE = 0;
    static final String API_URL = "http://"+"192.168.1.3"+":"+"5000"+"/";
    static final String FAIL_CONNECTION = "Failed to connect to server";
    static final String IMAGE_NAME = "image";
    static final String IMAGE_FILE_NAME = "androidFlask.jpg";
    static final String MEDIA_TYPE_PARSER = "image/*jpg";
    static final String INTENT_TYPE = "*/*";
    TextToSpeech voice;
    Button upload;
    Button camera;
    Bitmap cameraPhoto;
    ImageView image;
    TextView welcomeMessage;
    TextView caption;
    String selectedImagePath;
    boolean isCamera;
    boolean isUpload;

    @SuppressLint({"CutPasteId", "SetTextI18n"})
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        voice = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                int result = voice.setLanguage(Locale.ENGLISH);
                if (result == TextToSpeech.LANG_MISSING_DATA
                        || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.e("TTS", "Language not supported");
                } else {
                    caption.setEnabled(true);
                }
            } else {
                Log.e("TTS", "Initialization failed");
            }
        });

        caption = findViewById(R.id.caption_lbl);
        upload = findViewById(R.id.upload_btn);
        image = findViewById(R.id.loaded_image);
        welcomeMessage = findViewById(R.id.caption_lbl);
        camera = findViewById(R.id.camera_btn);
        isCamera = false;
        isUpload = false;


        ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.INTERNET}, 2);
        ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);


        upload.setOnClickListener(v -> {
            caption.setText("");
            Intent intent = new Intent();
            intent.setType(INTENT_TYPE);
            intent.setAction(Intent.ACTION_GET_CONTENT);
            startActivityForResult(intent, UPLOAD_ACTION_CODE);
            (new Handler()).postDelayed(this::caption, 5000);
        });

        camera.setOnClickListener(view -> {
            caption.setText("");
            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            if(intent.resolveActivity(getPackageManager()) != null){
                startActivityForResult(intent,CAMERA_ACTION_CODE);
            }
            else {
                Toast.makeText(getApplicationContext(),"NO APP SUPPORT THIS ACTION",Toast.LENGTH_LONG).show();
            }
        });


    }



    private void speak() {
        String text = caption.getText().toString();
        float pitch = (float) 50 / 50;
        if (pitch < 0.1) pitch = 0.1f;
        float speed = (float) 50/ 50;
        if (speed < 0.1) speed = 0.1f;
        voice.setPitch(pitch);
        voice.setSpeechRate(speed);
        voice.speak(text, TextToSpeech.QUEUE_FLUSH, null);
    }

    private void caption(){
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap. Config.RGB_565;
        Bitmap bitmap = null;
        if(isUpload){
            bitmap = BitmapFactory.decodeFile(selectedImagePath, options);
        }
        else if(isCamera){
            bitmap = cameraPhoto;
        }

        assert bitmap != null;
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
        byte[] byteArray = stream.toByteArray();
        RequestBody postBodyImage = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(IMAGE_NAME, IMAGE_FILE_NAME, RequestBody.create(MediaType.parse(MEDIA_TYPE_PARSER), byteArray))
                .build();
        postRequest(postBodyImage);
        (new Handler()).postDelayed(this::speak, 4000);
    }

    protected void onDestroy() {
        if (voice != null) {
            voice.stop();
            voice.shutdown();
        }
        super.onDestroy();
    }

    @Override
    protected void onActivityResult(int reqCode, int resCode, Intent data) {
        super.onActivityResult(reqCode, resCode, data);
        if (reqCode == UPLOAD_ACTION_CODE && resCode == RESULT_OK && data != null) {
            isUpload = true;
            Uri uri = data.getData();
            selectedImagePath = GetRealPath(getApplicationContext(), uri);
            image.setImageBitmap(BitmapFactory.decodeFile(selectedImagePath));
        }
        else if(reqCode == CAMERA_ACTION_CODE && resCode == RESULT_OK && data != null){
            isCamera = true;
            Bundle bundle = data.getExtras();
            cameraPhoto = (Bitmap) bundle.get("data");
            image.setImageBitmap(cameraPhoto);
            (new Handler()).postDelayed(this::caption, 1000);
        }
    }

    @Nullable
    public static String GetRealPath(@NonNull Context context, @NonNull Uri uri) {
        final ContentResolver contentResolver = context.getContentResolver();
        if (contentResolver == null){
            return null;
        }
        String filePath = context.getApplicationInfo().dataDir + File.separator + System.currentTimeMillis();
        File file = new File(filePath);
        try {
            InputStream inputStream = contentResolver.openInputStream(uri);
            if (inputStream == null){
                return null;
            }
            OutputStream outputStream = new FileOutputStream(file);
            byte[] buf = new byte[1024];
            int length;
            while ((length = inputStream.read(buf)) > 0){
                outputStream.write(buf, 0, length);
            }
            outputStream.close();
            inputStream.close();
        } catch (IOException ignore) {
            return null;
        }
        return file.getAbsolutePath();
    }


    void postRequest(RequestBody postBody) {
        OkHttpClient client = new OkHttpClient();
        Request request = new Request.Builder()
                .url(MainActivity.API_URL)
                .post(postBody)
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                call.cancel();
                runOnUiThread(() -> Toast.makeText(getApplicationContext(), FAIL_CONNECTION, Toast.LENGTH_LONG).show());
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull final Response response) {
                runOnUiThread(() -> {
                    try {

                        welcomeMessage.setText(response.body().string());
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                });
            }
        });
    }


}