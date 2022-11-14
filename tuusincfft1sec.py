import socket
import pickle
from tkinter import N


import time

import numpy as np
import matplotlib.pyplot as plt

kazu=64

zyusyo=5656

value_freq=50
MSGLEN =3856

t1 = 0/value_freq					# 解析対象データの開始時刻
t2 = t1 + kazu/value_freq
datatime = [0 for f in range(kazu)]
class MySocket:
    """demonstration class only
      - coded for clarity, not efficiency
    """
    

    def __init__(self, sock=None):
        
        print(MSGLEN)
        if sock is None:
            self.sock = socket.socket(
                            socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock

    def connect(self, host, port):
        self.sock.connect((host, port))

    def mysend(self, msg):
        totalsent = 0
        while totalsent < MSGLEN:
            sent = self.sock.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalsent = totalsent + sent

    def myreceive(self):
        chunks = []
        bytes_recd = 0
        while bytes_recd < MSGLEN:
            chunk = self.sock.recv(min(MSGLEN - bytes_recd, 2048))
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
        return b''.join(chunks)

volt = [[0 for f in range(8)]for i in range(kazu)]

class fftkeisan:
	
	SpectrumAmplitude = [[0.0 for f in range(kazu)]for i in range(8)]
	Freqency = [[0.0 for f in range(kazu)]for i in range(8)]

	def letsfft(ch,chnum,volt):
		for i in range(kazu):
			ch[i] = volt[i][chnum]
		chf= np.fft.fft(ch)
		SpectrumAmplitude = [0.0] *kazu
		Freqency = [0.0] *kazu
		for i in range(kazu):
			SpectrumAmplitude[i] = np.sqrt(
			chf[i].real * chf[i].real + chf[i].imag * chf[i].imag
			)
			Freqency[i] = (i * value_freq) / kazu
		fftkeisan.Freqency[chnum] = Freqency
		fftkeisan.SpectrumAmplitude[chnum] = SpectrumAmplitude
	

	def fftkiroku(chnum):
		Freqency= fftkeisan.Freqency[chnum]
		SpectrumAmplitude = fftkeisan.SpectrumAmplitude[chnum]
	
		fftkekka = [[0.0 for f in range(kazu)]for i in range(2)]
		fftkekka[0] = SpectrumAmplitude
		fftkekka[1] = Freqency
		"""f = open('out%d.csv'%(chnum+1),'a',)
		writer=csv.writer(f)
		writer.writerows(fftkekka)
		f.close()"""

		np.save('out%d.npy'%(chnum+1),fftkekka)
		print()
		
	def fftkunren(chnum):
		"""f = open('out%d.csv'%(chnum+1),'a',)
		fftkekka = [[0.0 for f in range(kazu)]for i in range(2)]
		reader=csv.reader(f)
		for row in reader:
			fftkekka[0] = "".join(row)
			fftkekka[1] = "".join(row)
		f.close()"""

		fftkekka=np.load('out%d.npy'%(chnum+1))
		plt.plot(fftkekka[1],fftkekka[0])
		plt.show()



		
	def plotfft(ch,chnum):
		SpectrumAmplitude = fftkeisan.SpectrumAmplitude[chnum]
		Freqency = fftkeisan.Freqency[chnum]
		global datatime
	
		for cnt in range(kazu):
			if cnt < kazu-1:
				datatime[cnt+1] = datatime[cnt] + 0.02					# 時刻データ：左端は０
		
		x = ch					# 振幅データ：左から二番目は１

		# 解析対象の関数の波形
		plt.subplot(2,1,1)
		plt.plot(datatime, x, color="b", linewidth=1.0, linestyle="-")
		plt.xlim(datatime[0], datatime[kazu-1])
		plt.ylim(-3.3, 3.3)
		plt.title("signal", fontsize=14, fontname='serif')
		plt.xlabel("Time [s]", fontsize=14, fontname='serif')
		plt.ylabel("Amplitude", fontsize=14, fontname='serif')

		# 振幅スペクトルと位相スペクトルの波形
		plt.subplot(2,1,2)
		plt.plot(Freqency, SpectrumAmplitude, color="b", linewidth=1.0, linestyle="-")
		plt.xlim(0, value_freq/2.0)			
		plt.ylim(0, 600.0)
		#plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])		# 目盛り
		#plt.yticks([0, 100, 200, 300, 400, 500, 600])		# 目盛り
		plt.grid(True) 						# gridを表示する
		plt.title("freqency1 spectrum", fontsize=14, fontname='serif')
		plt.xlabel("freqency1 [Hz]", fontsize=14, fontname='serif')
		plt.ylabel("Amplitude", fontsize=14, fontname='serif')
		
		
		# スペクトル値ピークの周波数を求める：直流成分を除く
		peakFreq = 0.0
		peakValue = 0.0
		for k in range(0, kazu):
			# 0.5Hz成分以上からナイキスト周波数まで
			if Freqency[k] > 0.5 and Freqency[k] < value_freq/2.0:
				# 値がこれまでの最大値より大きければ最大値情報を更新
				if SpectrumAmplitude[k] > peakValue:
					peakFreq = Freqency[k]
					peakValue = SpectrumAmplitude[k]
		
		# 結果を画面に出力する
		print("peak frequency : " + str(peakFreq) + " [Hz]")
		print("heart rate : " + str(60.0*peakFreq) + " [bpm: beat per minute]")
	
		
		# グラフを画面に表示する
		plt.draw()
		plt.pause(0.0001)
		plt.clf()


def connectdarui():
	byvolt = [[0 for f in range(8)]for i in range(kazu)]
	x = np.arange(0, 1.28, 1/50)
	y = np.floor(np.sin(2*np.pi*10*x)*2048)
	for cnt in range(kazu):
		for it in range(8):
			byvolt[cnt][it]=y[cnt]
	byvolt = pickle.dumps(byvolt)
	time.sleep(1.28)
	return byvolt
	
	
	



ch1=[0 for f in range(kazu)]
ch2=[0 for f in range(kazu)]
ch3=[0 for f in range(kazu)]
ch4=[0 for f in range(kazu)]
ch5=[0 for f in range(kazu)]
ch6=[0 for f in range(kazu)]
ch7=[0 for f in range(kazu)]
ch8=[0 for f in range(kazu)]



"""
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect(('192.168.11.13', zyusyo))
ms = MySocket(s)
"""
while True:
	cnt=0
#	byvolt = ms.myreceive()
	byvolt = connectdarui()

	iv = pickle.loads(byvolt)
	print(iv)
	for cnt in range(kazu):
		for it in range(8):
			ivint=int(iv[cnt][it])
			volt[cnt][it] = 5.0 * ivint / 4096
		
	
	#	print(volt[cnt][0])

	fftk= fftkeisan
	for it in range(8):
		fftk.letsfft(ch1,it,volt)
		fftk.fftkiroku(it)
	"""
	fftk.letsfft(ch2,1,volt)
	fftk.letsfft(ch3,2,volt)
	fftk.letsfft(ch4,3,volt)
	fftk.letsfft(ch5,4,volt)
	fftk.letsfft(ch6,5,volt)
	fftk.letsfft(ch7,6,volt)
	fftk.letsfft(ch8,7,volt)
"""
	fftk.fftkunren(0)



