import numpy as np

snr_array = np.array([1.95, 4, 6, 8, 10, 11.95, 14.05, 16, 17.9, 19.9, 21.5, 23.45, 25.0, 27.30, 29, np.Inf])

def snr_to_cqi(snr):
	# SNR-to-CQI conversion, got from "Downlink SNR to CQI Mapping for Different Multiple Antenna Techniques in LTE", Table III.
	snr_to_cqi_table = snr_array
	cqi = np.where(snr_to_cqi_table > snr)[0][0]
	return cqi

def cqi_to_coderate(cqi):
	cqi_to_coderate_table = np.array([0, 0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547])
	if cqi < 16:
		return cqi_to_coderate_table[cqi]
	else:
		print('Error cqi_to_coderate')
		return -1

def tbs_idx_from_mcs(mcs):
	tbs_idx_from_mcs_table = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
	if mcs < 29:
		return tbs_idx_from_mcs_table[mcs]
	else:
		print('Error tbs_idx_from_mcs')
		return -1

def tbs_from_idx(tbs_idx): # We assume n_prb = 50
	tbs_table = np.array([1384, 1800, 2216, 2856, 3624, 4392, 5160, 6200, 6968, 7992, 8760, 9912, 11448, 12960, 14112, 15264, 16416, 18336, 19848, 21384, 22920, 25456, 27376, 28336, 30576, 31704, 36696])
	# tbs_table = np.array([16, 24, 32, 40, 56, 72, 88, 104, 120, 136, 144, 176, 208, 224, 256, 280, 328, 336, 376, 408, 440, 488, 520, 552, 584, 616, 712])
	if tbs_idx < 27:
		return tbs_table[tbs_idx] # bits
	else:
		print('Error tbs_from_idx')
		return -1

def coderate_funct(tbs):
	CP_NSYMB = 7 # 6 normal, 7 ext.
	NRE = 12
	n_prb = 50
	nof_re = (2 * (CP_NSYMB - 1)) * n_prb * NRE
	return (tbs + 24) / nof_re

def qm_from_mcs(mcs):
	if mcs <= 17 or mcs == 30:
		return 2  #QPSK
	elif mcs <= 17 or mcs == 30:
		return 4  #16QAM
	else:
		return 6  #64QAM

def cqi_to_mcs_tbs(cqi):
	# max_mcs = 18
	# max_Qm = 4 # Allow 16-QAM in PUSCH Only
	max_mcs = 26
	max_Qm = 6

	sel_mcs = max_mcs + 1
	max_coderate = cqi_to_coderate(int(cqi))

	while True:
		sel_mcs -= 1
		tbs_idx = tbs_idx_from_mcs(sel_mcs)
		tbs = tbs_from_idx(tbs_idx)
		coderate = coderate_funct(tbs)
		Qm = np.min([max_Qm, qm_from_mcs(sel_mcs)])
		eff_coderate = coderate/Qm

		if not ((sel_mcs > 0 and coderate > max_coderate) or (eff_coderate > 0.930)):
			break
	return sel_mcs, tbs

def snr_to_mcs(snr): #return the maximum mcs value allowed by the snr
	cqi = snr_to_cqi(snr)
	mcs, _ = cqi_to_mcs_tbs(cqi)
	return mcs

def mcs_to_tbs(mcs): #return the tbs in bits
	tbs_idx = tbs_idx_from_mcs(mcs)
	tbs = tbs_from_idx(tbs_idx)
	return tbs

class SNRtoTBS:
	def __init__(self):
		tbs_values = []
		for snr in snr_array:
			if snr == np.Inf:
				snr = 50
			mcs = snr_to_mcs(snr*0.99)
			tbs = np.int(np.round(mcs_to_tbs(mcs)/50))
			tbs_values.append(tbs)
		self.tbs_values = np.array(tbs_values, dtype = np.int32)

	def snr_to_tbs(self, snr):
		index = np.where(snr_array >= snr)[0][0]
		return self.tbs_values[index]


if __name__ == '__main__':
	map = SNRtoTBS()
	snr_values = np.linspace(0,35,20)
	for snr in snr_values:
	# for snr in snr_array:
		if snr < np.Inf:
			mcs = snr_to_mcs(snr)
			tbs = np.int(np.round(mcs_to_tbs(mcs)/50))
			print('snr = {}, mcs = {}, tbs = {}'.format(snr, mcs, tbs))

			tbs = map.snr_to_tbs(snr)
			print('map object tbs = {}'.format(tbs))
