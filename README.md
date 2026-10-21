# Simulación de Colisión de Galaxias con el Problema de N Cuerpos

Simulador de la colisión de dos galaxias de disco mediante el problema gravitacional de N cuerpos (O(N²)), integrado con el método leapfrog y fuerzas suavizadas. El proyecto implementa el cálculo de aceleraciones en tres versiones (NumPy, Numba multihilo en CPU y Numba CUDA en GPU con memoria compartida) y compara su rendimiento: hasta ~86× de aceleración en CPU frente a NumPy y ~56× de la GPU frente a la CPU paralela, con hasta 64 000 partículas. Incluye validación física (convergencia en órbitas keplerianas, conservación de energía y de momentos lineal y angular), generación de condiciones iniciales realistas, visualización en video y una versión interactiva en el navegador.


https://github.com/user-attachments/assets/a2e33f16-a977-495d-949b-618d1e7ff95d

<img width="1280" height="720" alt="colision_final" src="https://github.com/user-attachments/assets/837d820b-0fd7-424f-8c3c-5ad29fcc9bec" />
