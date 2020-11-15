#from __future__ import (division, absolute_import, print_function, unicode_literals)


import datetime
import math
import cv2
import numpy as np
from datetime import datetime


# variaveis globais
width = 0
height = 0
ContadorEntradas = 0
ContadorSaidas = 0

now = datetime.now() # estimativa o tempo de processamento
hora = now.hour
minuto = now.minute
segundo = now.second

# este valor eh empirico. Ajuste-o conforme sua necessidade
AreaContornoLimiteMin = 3000
ThresholdBinarizacao = 70  # este valor eh empirico, Ajuste-o conforme sua necessidade
#Distância do certo para as linha
OffsetLinhasRef_azul = 170
OffsetLinhasRef_vermelho = 100

#############################
#Carrega a biblioteca YOLO

print("LOADING YOLO")
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()

output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("YOLO LOADED")

##########################################################


# Verifica se o corpo detectado esta entrando da zona monitorada
def TestaInterseccaoEntrada(y, CoordenadaYLinhaEntrada, CoordenadaYLinhaSaida):
    DiferencaAbsoluta = abs(y - CoordenadaYLinhaEntrada)

    if ((DiferencaAbsoluta <= 2) and (y < CoordenadaYLinhaSaida)):
        return 1
    else:
        return 0



# Verifica se o corpo detectado esta saindo da zona monitorada
def TestaInterseccaoSaida(y, CoordenadaYLinhaEntrada, CoordenadaYLinhaSaida):
    DiferencaAbsoluta = abs(y - CoordenadaYLinhaSaida)

    if ((DiferencaAbsoluta <= 2) and (y > CoordenadaYLinhaEntrada)):
        return 1
    else:
        return 0






def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


#Entreda do video
camera = cv2.VideoCapture("peq.mp4")



# forca a camera a ter resolucao 640x480
camera.set(3, 640)
camera.set(4, 480)


# faz algumas leituras de frames antes de consierar a analise
# motivo: algumas camera podem demorar mais para se "acosumar a luminosidade" quando ligam, capturando frames consecutivos com muita variacao de luminosidade. Para nao levar este efeito ao processamento de imagem, capturas sucessivas sao feitas fora do processamento da imagem, dando tempo para a camera "se acostumar" a luminosidade do ambiente
# esse for faz isso
for i in range(0, 20):
    (grabbed, Frame) = camera.read()


while True:


    Frame = np.hstack((Frame, white_balance(Frame)))

    # le primeiro frame 
    (grabbed, Frame) = camera.read()

    # se nao foi possivel obter frame, nada mais deve ser feito
    if not grabbed:
        break

    ###############
    #determina resolucao da imagem
    height, width, channels = Frame.shape
    
    # preprocessa as imagens usando blob
    blob = cv2.dnn.blobFromImage(Frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    #Detecta os objetos na imagem
    net.setInput(blob)
    outs = net.forward(output_layers)

    
    class_ids = []
    confidences = []
    boxes = []
    
    ####################################

   ##POSSO CONTAR AS PESSOAS CONFORME ELAS PASSAM DA LINHA AZUL PARA VEMELHA OU VERMELHA PARA AZUL

    #Conta quantos pessoas estão na imagem
    QtdeContornos = 0
    
    # desenha linhas de referencia
    CoordenadaYLinhaEntrada = (int((height / 2)-OffsetLinhasRef_azul))
    CoordenadaYLinhaSaida = (int((height / 2)+OffsetLinhasRef_vermelho))
    cv2.line(Frame, (0, CoordenadaYLinhaEntrada), (width, CoordenadaYLinhaEntrada), (255, 0, 0), 2)
    cv2.line(Frame, (0, CoordenadaYLinhaSaida), (width,CoordenadaYLinhaSaida), (0, 0, 255), 2)

    
    ################
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Objeto detectado
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # cordenadas do retangulo
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = [254.0, 244.0, 70.0]

            if(label == 'person'):
                cv2.rectangle(Frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(Frame, label, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX,1/2, color, 2)

                # determina o ponto central do contorno e desenha um circulo para indicar
                CoordenadaXCentroContorno = (int((x+x+w)/2))
                CoordenadaYCentroContorno = (int((y+y+h)/2))
                PontoCentralContorno = (CoordenadaXCentroContorno, CoordenadaYCentroContorno)
                cv2.circle(Frame, PontoCentralContorno, 1, (0, 0, 0), 5)

                # Para fins de depuracao, contabiliza numero de contornos encontrados
                QtdeContornos = QtdeContornos+1

                # testa interseccao dos centros dos contornos com as linhas de referencia
                # dessa forma, contabiliza-se quais contornos cruzaram quais linhas !!!(num determinado sentido)!!!
                if (TestaInterseccaoEntrada(CoordenadaYCentroContorno, CoordenadaYLinhaEntrada,CoordenadaYLinhaSaida)):
                    ContadorEntradas += 1

                if (TestaInterseccaoSaida(CoordenadaYCentroContorno, CoordenadaYLinhaEntrada,CoordenadaYLinhaSaida)):  
                    ContadorSaidas += 1

    
    #########################

    print ("Contornos encontrados: "+str(QtdeContornos))

    # Escreve na imagem o numero de pessoas que entraram ou sairam da area vigiada
    cv2.putText(Frame, "Entradas: {}".format(str(ContadorEntradas)), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 1), 2)
    cv2.putText(Frame, "Saidas: {}".format(str(ContadorSaidas)), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Original", Frame)
    cv2.waitKey(1)


camera.release()
cv2.destroyAllWindows()

#tempo de execução
now = datetime.now()
print("Tempo de execução: ",(now.hour - hora),":",(now.minute - minuto),":",abs(now.second - segundo))
