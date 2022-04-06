package ml.imagerecognitionapp

import android.Manifest
import android.app.Activity
import android.app.Dialog
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat

import kotlinx.android.synthetic.main.activity_main.*
import ml.imagerecognitionapp.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    lateinit var dialog:Dialog
    lateinit var bitmap: Bitmap
    lateinit var wordlist:List<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        var filename="label.txt"
        var inputstring=application.assets.open(filename).bufferedReader().use { it.readText() }
        var wordlist=inputstring.split("\n")

        btn_take_picture.isEnabled=false

        if(ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)
        }
        else
        {
            btn_take_picture.isEnabled=true
        }

        btn_take_picture.setOnClickListener{

            dialog= Dialog(this@MainActivity)
            dialog.setContentView(R.layout.layout_img_select)

            dialog.window!!.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
            dialog.show()

            var cam:ImageView=dialog.findViewById(R.id.cam)
            var file:ImageView=dialog.findViewById(R.id.file)
            var camtext:TextView=dialog.findViewById(R.id.ca)
            var filetext:TextView=dialog.findViewById(R.id.fil)

            cam.setOnClickListener{
               takepic()
                dialog.dismiss()
            }
            camtext.setOnClickListener{
                takepic()
                dialog.dismiss()
            }

            file.setOnClickListener{
                selectfile()
                dialog.dismiss()
            }
            filetext.setOnClickListener {
                selectfile()
                dialog.dismiss()
            }

        }

    }
    private fun takepic() {
        var i = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        try {

            startActivityForResult(i, 100)
        } catch (e: ActivityNotFoundException) {
            Log.d("Error", e.printStackTrace().toString())
        }
    }

    private fun selectfile(){
        var i=Intent(Intent.ACTION_GET_CONTENT)
        i.type="image/*"

        startActivityForResult(i,200)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if(requestCode==100 && grantResults[0]==PackageManager.PERMISSION_GRANTED)
        {
                btn_take_picture.isEnabled=true
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)



        if (requestCode == 100 && resultCode==Activity.RESULT_OK){
            val pic:Bitmap?=data!!.getParcelableExtra("data")
            if(pic==null)
                img.setImageURI(data!!.data)
            else
                img.setImageBitmap(pic)
                bitmap=MediaStore.Images.Media.getBitmap(this.contentResolver,data!!.data)
            predict()
        }
        else if(requestCode == 200 && resultCode==Activity.RESULT_OK)
        {
                img.setImageURI(data!!.data)
            bitmap=MediaStore.Images.Media.getBitmap(this.contentResolver,data!!.data)
            predict()
        }

    }

    private fun predict()
    {
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)

        val predict:Bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val model = MobilenetV110224Quant.newInstance(this)
        var pBuffer=TensorImage.fromBitmap(predict)
        var bBuffer = pBuffer.buffer

        inputFeature0.loadBuffer(bBuffer)

        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        var max=getmax(outputFeature0.floatArray)
        txt_result.text=wordlist[max]
        model.close()
    }

    fun getmax(arr:FloatArray):Int
    {
        var id=0
        var min=0.0f
        for(i in 0..1000){
            if(arr[i]>min)
            {
                id=i
                min=arr[i]

            }
        }
        return id
    }
}