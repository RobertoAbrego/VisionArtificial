
package com.roberto.vision;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.Map;

import javax.imageio.ImageIO;

@RestController
public class ClasificadorController {

    @Value("${MODELO_RUTA:${modelo.ruta:modelo/modeloDenso.onnx}}")
    private String rutaModelo;
    private OrtEnvironment entorno;
    private OrtSession sesion;
    private String nombreEntrada;

    @PostConstruct
    public void cargarModelo() throws Exception {
        entorno = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions opciones =
            new OrtSession.SessionOptions();

        try {
            sesion = entorno.createSession(
                Path.of(rutaModelo).toAbsolutePath().toString(),
                opciones
            );
        } finally {
            opciones.close();
        }

        if (sesion.getInputNames().size() != 1) {
            throw new IllegalStateException(
                "Se esperaba una sola entrada en el modelo ONNX."
            );
        }

        nombreEntrada = sesion.getInputNames()
                              .iterator()
                              .next();

        System.out.println("Modelo ONNX cargado: " + rutaModelo);
        System.out.println("Entrada: " + nombreEntrada);
        System.out.println("Salidas: " + sesion.getOutputNames());
    }

    @PostMapping(
        value = "/api/clasificar",
        consumes = MediaType.MULTIPART_FORM_DATA_VALUE
    )
    public Map<String, Object> clasificar(
        @RequestParam("imagen") MultipartFile archivo
    ) throws Exception {

        if (archivo.isEmpty()) {
            throw new ResponseStatusException(
                HttpStatus.BAD_REQUEST,
                "Debes seleccionar una imagen."
            );
        }

        BufferedImage original;

        try {
            original = ImageIO.read(archivo.getInputStream());
        } catch (IOException e) {
            throw new ResponseStatusException(
                HttpStatus.BAD_REQUEST,
                "No se pudo leer la imagen.",
                e
            );
        }

        if (original == null) {
            throw new ResponseStatusException(
                HttpStatus.BAD_REQUEST,
                "El archivo no es una imagen compatible."
            );
        }

        float[] pixeles = prepararImagen(original);

        // Entrada NHWC: lote, alto, ancho, canales.
        long[] dimensiones = {1, 100, 100, 1};

        try (OnnxTensor tensor = OnnxTensor.createTensor(
                 entorno,
                 FloatBuffer.wrap(pixeles),
                 dimensiones
             );
             OrtSession.Result resultado = sesion.run(
                 Map.of(nombreEntrada, tensor)
             )) {

            Object valorSalida = resultado.get(0).getValue();

            if (!(valorSalida instanceof float[][] salida)
                    || salida.length == 0
                    || salida[0].length == 0) {
                throw new IllegalStateException(
                    "La salida ONNX no tiene la forma esperada [1, 1]."
                );
            }

            float puntuacionPerro = salida[0][0];

            if (!Float.isFinite(puntuacionPerro)
                    || puntuacionPerro < 0.0f
                    || puntuacionPerro > 1.0f) {
                throw new IllegalStateException(
                    "La salida no parece ser una probabilidad entre 0 y 1."
                );
            }

            String clase = puntuacionPerro >= 0.5f
                ? "Perro"
                : "Gato";

            return Map.of(
                "clase", clase,
                "puntuacionPerro", puntuacionPerro
            );
        }
    }

    private float[] prepararImagen(BufferedImage original) {
        BufferedImage gris = new BufferedImage(
            original.getWidth(),
            original.getHeight(),
            BufferedImage.TYPE_BYTE_GRAY
        );

        Graphics2D graficosGrises = gris.createGraphics();
        graficosGrises.drawImage(original, 0, 0, null);
        graficosGrises.dispose();

        BufferedImage redimensionada = new BufferedImage(
            100,
            100,
            BufferedImage.TYPE_BYTE_GRAY
        );

        Graphics2D graficos = redimensionada.createGraphics();

        graficos.setRenderingHint(
            RenderingHints.KEY_INTERPOLATION,
            RenderingHints.VALUE_INTERPOLATION_BILINEAR
        );

        graficos.drawImage(gris, 0, 0, 100, 100, null);
        graficos.dispose();

        float[] pixeles = new float[100 * 100];

        for (int y = 0; y < 100; y++) {
            for (int x = 0; x < 100; x++) {
                int rgb = redimensionada.getRGB(x, y);
                int intensidad = rgb & 0xFF;

                pixeles[y * 100 + x] =
                    intensidad / 255.0f;
            }
        }

        return pixeles;
    }

    @PreDestroy
    public void cerrarModelo() throws Exception {
        if (sesion != null) {
            sesion.close();
        }

        // OrtEnvironment es compartido por la aplicación;
        // no se cierra aquí.
    }
}
