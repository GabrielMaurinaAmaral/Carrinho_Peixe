import cv2

COR_VERMELHO = (0, 0, 255)
COR_VERDE = (0, 255, 0)
COR_AZUL = (255, 0, 0)

class Deteccao_Preto:

    def __init__(self):
        # Tamanho do kernel para operações de processamento de imagem
        self.kernel_size = (5, 5)
        # Intervalo de cores HSV para a detecção da cor do peixe
        self.range_cor = [(0, 0, 0), (255, 50, 50)]

    def linearizar_frame(self, frame):
        # Converte o frame de BGR para HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Cria uma máscara preta com base no intervalo de cores definido
        black_mask = cv2.inRange(hsv, self.range_cor[0], self.range_cor[1])
        # Aplica a máscara ao frame original
        black_result = cv2.bitwise_and(frame, frame, mask=black_mask)
        # Converte o resultado para tons de cinza
        gray_frame = cv2.cvtColor(black_result, cv2.COLOR_BGR2GRAY)
        #
        suavisacao_frame = cv2.medianBlur(gray_frame, 5)  
        # Converte a imagem em um formato binário
        _, frame_binario = cv2.threshold(suavisacao_frame, 0, 255, cv2.THRESH_BINARY)
        # Cria um elemento estruturante em forma de cruz
        pixel_preenchimento = cv2.getStructuringElement(cv2.MORPH_CROSS, self.kernel_size)
        # Aplica uma operação morfológica para fechar pequenos buracos na imagem
        frame = cv2.morphologyEx(frame_binario, cv2.MORPH_CLOSE, pixel_preenchimento)
        return frame

    def trancking_peixe(self, frame, frame_binario):
        # Encontra os contornos na imagem binária
        contornos, _ = cv2.findContours(frame_binario, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if contornos:
            # Encontra o maior contorno com base na área
            max_Area = cv2.contourArea(contornos[0])
            contorno_Max_Area_Id = 0
            for i, contorno in enumerate(contornos):
                if max_Area < cv2.contourArea(contorno):
                    max_Area = cv2.contourArea(contorno)
                    contorno_Max_Area_Id = i
            maior_Contorno = contornos[contorno_Max_Area_Id]
            # Desenha o contorno e a grade
            self.desenha_trancking(frame, maior_Contorno)

    def desenha_trancking(self, frame, contorno):
        # Encontra o retângulo convexo em torno do contorno
        hull = cv2.convexHull(contorno)
        # Desenha o retângulo convexo no frame
        cv2.drawContours(frame, [hull], -1, COR_AZUL, 1)
        # Encontra o centro do retângulo convexo (pode ser usado como ponto central do contorno)
        M = cv2.moments(hull)
        if M["m00"] != 0:
            x_Central = int(M["m10"] / M["m00"])
            y_Central = int(M["m01"] / M["m00"])
        else:
            # Lide com o caso em que M["m00"] é zero (por exemplo, não há contorno válido)
            # Você pode definir x_Central e y_Central como valores padrão ou tomar alguma outra ação apropriada
            x_Central = 0
            y_Central = 0
        # Desenha um ponto no centro
        cv2.circle(frame, (x_Central, y_Central), 5, COR_VERMELHO, -1)
        # Desenha o contorno
        cv2.drawContours(frame, [contorno], -1, (0, 255, 255), 1)
        # Adicionar texto de posição
        texto = f"X: {x_Central}, Y: {y_Central}"
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)


    def desenha_grade(self, frame):
        # Obtém as dimensões do frame
        h_Frame, w_Frame, _ = frame.shape
        divisoes = 3  # Número de divisões ao longo de cada eixo (grade 3x3)
        # Desenha as linhas horizontais da grade
        for i in range(1, divisoes):
            y = int(i * h_Frame / divisoes)
            cv2.line(frame, (0, y), (w_Frame, y), COR_VERDE, 2)
        # Desenha as linhas verticais da grade
        for i in range(1, divisoes):
            x = int(i * w_Frame / divisoes)
            cv2.line(frame, (x, 0), (x, h_Frame), COR_VERDE, 2)

    def loop(self):
        webcam = cv2.VideoCapture(0)
        # Configura as propriedades da webcam
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        webcam.set(cv2.CAP_PROP_FPS, 30)

        while True:
            valido, frame = webcam.read()
            if valido:
                self.desenha_grade(frame)
                frame_binario = self.linearizar_frame(frame)
                self.trancking_peixe(frame, frame_binario)
                # Exibe os frames
                cv2.imshow('BINARIO', frame_binario)
                cv2.imshow('PRINCIPAL', frame)
                # Encerra o loop quando uma tecla for pressionada
                if cv2.waitKey(1) != -1:
                    break
            else:
                break

        # Libera a webcam e fecha as janelas
        webcam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    tracker = Deteccao_Preto()
    tracker.loop()
