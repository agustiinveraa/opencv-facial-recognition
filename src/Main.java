import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class Main {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Cargar el clasificador Haar para detección de rostros
        CascadeClassifier faceDetector = new CascadeClassifier();
        faceDetector.load("haarcascade_frontalface_alt.xml");

        // Inicializar la captura de la webcam
        VideoCapture capture = new VideoCapture(0); // Cambia el número a 1 si tienes múltiples cámaras

        if (!capture.isOpened()) {
            System.out.println("Error al abrir la webcam");
            return;
        }

        // Capturar y procesar cada fotograma de la webcam
        Mat frame = new Mat();
        while (true) {
            // Capturar un fotograma de la webcam
            capture.read(frame);

            // Convertir el fotograma a escala de grises
            Mat grayFrame = new Mat();
            Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
            Imgproc.equalizeHist(grayFrame, grayFrame);

            // Detectar caras en el fotograma
            MatOfRect faceDetections = new MatOfRect();
            faceDetector.detectMultiScale(grayFrame, faceDetections);

            // Dibujar rectángulos alrededor de las caras detectadas
            for (Rect rect : faceDetections.toArray()) {
                Imgproc.rectangle(frame, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                        new Scalar(0, 255, 0), 2);
            }

            // Mostrar el fotograma con las caras detectadas
            HighGui.imshow("DETECTOR FACIAL", frame);

            // Esperar 30 milisegundos antes de procesar el siguiente fotograma
            if (HighGui.waitKey(30) >= 0) {
                break;
            }
        }

        // Liberar la captura de la webcam y cerrar las ventanas
        capture.release();
        HighGui.destroyAllWindows();
    }
}
