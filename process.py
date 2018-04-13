#coding=utf-8

from queue import Queue
import pickle
import numpy as np
import json
import codecs
import unicodedata
import nltk
import sys
import os
from params import Params as hps
from tqdm import tqdm
from six.moves import xrange
import shutil
from random import shuffle

class data_loader(object):
	def __init__(self, use_pretrained=None):
		self.c_dict = {"_UNK": 0}
		self.w_dict = {"_UNK": 0}
		self.w_occurence = 0
		self.c_occurence = 0
		self.w_count = 1
		self.c_count = 1
		self.w_unknown_count = 0
		self.c_unknown_count = 0
		self.invalid_q = 0

		if use_pretrained:
			self.w_dict, self.w_count = self.process_glove(hps.glove_dir, self.w_dict, self.w_count, hps.emb_size)
			self.ids2char = {v: k for k, v in self.c_dict.items()}
			self.c_dict, self.c_count = self.process_glove(hps.glove_char, self.c_dict, self.c_count, 300)
			self.ids2char = {v: k for k, v in self.c_dict.items()}

	def ind2word(self, ids):
		output = []
		for i in ids:
			output.append(str(self.ids2word[i]))
		return " ".join(output)

	def ind2char(self, ids):
		output = []
		for i in ids:
			for j in i:
				output.append(str(self.ids2char[j]))
			output.append(" ")
		return "".join(output)

	def process_glove(self, wordvecs, dict_, count, emb_size):
		print("Reading GloVe from: {}".format(wordvecs))
		with codecs.open(wordvecs, "rb", "utf-8") as f:
			line = f.readline()
			i = 0
			while line:
				vocab = line.split(" ")
				if len(vocab) != emb_size + 1:
					line = f.readline()
					continue
				vocab = normalize_text(''.join(vocab[0]))
				if vocab not in dict_:
					dict_[vocab] = count
				line = f.readline()
				count += 1
				i += 1
				if i % 100 == 0:
					sys.stdout.write("\rProcessing line %d       " % i)
			print("")
		print('word vocab size: %d' % count)
		return dict_, count

	def process_json(self, file_dir, out_dir, write_=True):
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		self.data = json.load(codecs.open(file_dir, "rb", "utf-8"))
		self.loop(self.data, out_dir, write_=write_)
		with codecs.open("dictionary.txt", "wb", "utf-8") as f:
			for key, value in sorted(self.w_dict.items(), key=lambda x: x[1]):
				f.write("%s: %s" % (key, value) + "\n")

	def loop(self, data, dir_=hps.train_dir, write_=True):
		for topic in tqdm(data['data'], total=len(data['data'])):
			for para in topic['paragraphs']:
				words_c, chars_c = self.add_to_dict(para['context'].lower())
				if len(words_c) >= hps.max_c_len:
					continue
				for qas in para['qas']:
					question = qas['question']
					words, chars = self.add_to_dict(question.lower())
					if len(words) >= hps.max_q_len:
						continue
					pre_ans = []
					for ans in qas['answers']:
						ans_ids, _ = self.add_to_dict(ans['text'].lower())
						ans_lens = len(ans_ids)
						ans_start_pos = ans['answer_start']
						answer_repeat = False
						for pre_ans_start_pos, pre_ans_lens in pre_ans:
							if ans_start_pos == pre_ans_start_pos and ans_lens == pre_ans_lens:
								answer_repeat = True
								break
						if answer_repeat:
							continue
						pre_ans.append((ans_start_pos, ans_lens))
						answers = find_answer_index(words_c, ans_ids)
						for answer in answers:
							start_i, finish_i = answer
							if start_i == -1:
								self.invalid_q += 1
								continue
							if write_:
								write_file([str(start_i), str(finish_i)], dir_ + hps.target_dir)
								write_file(words, dir_ + hps.q_word_dir)
								write_file(chars, dir_ + hps.q_chars_dir)
								write_file(words_c, dir_ + hps.c_word_dir)
								write_file(chars_c, dir_ + hps.c_chars_dir)

	def process_word(self, line):
		for word in line:
			word = word.replace(" ", "").strip()
			word = normalize_text(''.join(word))
			if word:
				if word not in self.w_dict:
					self.w_dict[word] = self.w_count
					self.w_count += 1

	def add_to_dict(self, line):
		splitted_line = nltk.word_tokenize(line)
		self.process_word(splitted_line)

		words = []
		chars = []
		for i, word in enumerate(splitted_line):
			word = word.replace(" ", "").strip()
			word = normalize_text(''.join(word))
			if word:
				if i > 0:
					chars.append("_SPC")
				for char in word:
					char = self.c_dict.get(char, self.c_dict["_UNK"])
					chars.append(str(char))
					self.c_occurence += 1
					if char == 0:
						self.c_unknown_count += 1

				word = self.w_dict.get(word.strip().strip(" "), self.w_dict["_UNK"])
				words.append(str(word))
				self.w_occurence += 1
				if word == 0:
					self.w_unknown_count += 1
		return (words, chars)

def load_glove(dir_, name, vocab_size):
	emb_size = hps.emb_size if name != 'glove_char' else hps.char_emb_size
	glove = np.zeros((vocab_size, emb_size), dtype=np.float32)
	with codecs.open(dir_, "rb", "utf-8") as f:
		line = f.readline()
		i = 1
		while line:
			if i % 100 == 0:
				sys.stdout.write("\rProcessing %d vocabs       "%i)
			vector = line.split(" ")
			if len(vector) != hps.emb_size + 1:
				line = f.readline()
				continue
			name_ = vector[0]
			vector = vector[1:emb_size+1]
			if vector:
				try:
					vector = [float(n) for n in vector]
				except:
					assert 0
				vector = np.asarray(vector, np.float32)
				try:
					glove[i] = vector
				except:
					assert 0
			line = f.readline()
			i += 1
	print("\n")
	glove_map = np.memmap(hps.data_dir + name + ".np", dtype='float32', mode='write', shape=(vocab_size, emb_size))
	glove_map[:] = glove
	del glove_map

def reduce_glove(dir_, dict_):
	glove_f = []
	with codecs.open(dir_, "rb", "utf-8") as f:
		line = f.readline()
		i = 0
		while line:
			i += 1
			if i % 100 == 0:
				sys.stdout.write("\rProcessing %d vocabs       "%i)
			vector = line.split(" ")
			if len(vector) != hps.emb_size + 1:
				line = f.readline()
				continue
			vocab = normalize_text(''.join(vector[0:-hps.emb_size]))
			if vocab not in dict_:
				line = f.readline()
				continue
			glove_f.append(line)
			line = f.readline()
	print("\nTotal number of lines: {}\nReduced vocab size: {}".format(i, len(glove_f)))
	with codecs.open(dir_, "wb", "utf-8") as f:
		for line in glove_f[:-1]:
			f.write(line)
		f.write(glove_f[-1].strip("\n"))

def find_answer_index(context, answer):
	window_len = len(answer)
	answers = []
	if window_len == 1:
		indices = [i for i, ctx in enumerate(context) if ctx == answer[0]]
		for i in indices:
			answers.append((i, i))
		if not indices:
			answers.append((-1, -1))
		return answers
	for i in range(len(context)):
		if context[i:i+window_len] == answer:
			answers.append((i, i + window_len))
	if len(answers) == 0:
		return [(-1, -1)]
	else:
		return answers

def normalize_text(text):
	return unicodedata.normalize('NFD', text)

def write_file(indices, dir_, separate = "\n"):
	with codecs.open(dir_, "ab", "utf-8") as f:
		f.write(" ".join(indices) + separate)

def pad_data(data, max_word):
	padded_data = np.zeros((len(data), max_word), dtype=np.int32)
	for i, line in enumerate(data):
		for j, word in enumerate(line):
			if j >= max_word:
				break
			padded_data[i, j] = word
	return padded_data

def pad_char_data(data, max_char, max_words):
	padded_data = np.zeros((len(data), max_words, max_char), dtype=np.int32)
	for i, line in enumerate(data):
		for j, word in enumerate(line):
			if j >= max_words:
				break
			for k, char in enumerate(word):
				if k >= max_char:
					# ignore the rest of the word if it's longer than the limit
					break
				padded_data[i, j, k] = char
	return padded_data

def load_target(dir):
	data = []
	with codecs.open(dir, "rb", "utf-8") as f:
		line = f.readline()
		while line:
			line = [int(w) for w in line.split()]
			data.append(line)
			line = f.readline()
	return data

def load_word(dir):
	data = []
	w_len = []
	with codecs.open(dir, "rb", "utf-8") as f:
		line = f.readline()
		while line:
			line = [int(w) for w in line.split()]
			data.append(line)
			w_len.append(len(line))
			line = f.readline()
	return data, w_len

def load_char(dir):
	data = []
	w_len = []
	c_len_ = []
	with codecs.open(dir, "rb", "utf-8") as f:
		line = f.readline()
		while line:
			c_len = []
			chars = []
			line = line.split("_SPC")
			for word in line:
				c = [int(w) for w in word.split()]
				c_len.append(len(c))
				chars.append(c)
			data.append(chars)
			line = f.readline()
			c_len_.append(c_len)
			w_len.append(len(c_len))
	return data, c_len_, w_len

def data_queue(context_word_dir, question_word_dir, context_chars_dir, question_chars_dir, target_dir):
	_data_queue = Queue(maxsize=-1)
	data = []
	with codecs.open(context_word_dir, 'r', 'utf-8') as c_w_f:
		with codecs.open(question_word_dir, 'r', 'utf-8') as q_w_f:
			with codecs.open(context_chars_dir, 'r', 'utf-8') as c_c_f:
				with codecs.open(question_chars_dir, 'rb', 'utf-8') as q_c_f:
					with codecs.open(target_dir, 'r', 'utf-8') as t_f:
						c_word, q_word, c_chars, q_chars, target = c_w_f.readline().split(), q_w_f.readline().split(), \
								c_c_f.readline().split('_SPC'), q_c_f.readline().split('_SPC'), t_f.readline().split()
						while c_word and q_word and c_chars and q_chars and target:
							assert len(target) == 2
							assert len(c_word) == len(c_chars)
							assert len(q_word) == len(q_chars)
							splited_c_chars, splited_q_chars = [], []
							for c_c in c_chars:
								splited_c_chars.append(c_c.split())
							for q_c in q_chars:
								splited_q_chars.append(q_c.split())
							assert len(c_word) == len(splited_c_chars)
							assert len(q_word) == len(splited_q_chars)
							data.append((c_word, q_word, splited_c_chars, splited_q_chars, target))

							c_word, q_word, c_chars, q_chars, target = c_w_f.readline().split(), q_w_f.readline().split(), \
								c_c_f.readline().split('_SPC'), q_c_f.readline().split('_SPC'), t_f.readline().split()
	shuffle(data)
	for x in data:
		_data_queue.put(x)
	print('data queue ok!!!')
	return _data_queue

def get_batch(data):
	count = 0
	batch_c_word, batch_q_word, batch_c_chars, batch_q_chars, batch_target, batch_c_lens, batch_q_lens = \
																				[], [], [], [], [], [], []
	while count < hps.batch_size:
		c_word, q_word, c_chars, q_chars, target = data.get()
		data.put((c_word, q_word, c_chars, q_chars, target))

		c_lens = len(c_word)
		q_lens = len(q_word)

		batch_c_word.append(c_word)
		batch_q_word.append(q_word)
		batch_c_chars.append(c_chars)
		batch_q_chars.append(q_chars)
		batch_target.append(target)
		batch_c_lens.append(c_lens)
		batch_q_lens.append(q_lens)
		count += 1

	char_length = []
	for x in batch_c_chars + batch_q_chars:
		for c in x:
			char_length.append(len(c))
	max_char_length = max(char_length)
	max_c_length = max(batch_c_lens)
	max_q_length = max(batch_q_lens)

	_batch_c_word, _batch_q_word, _batch_c_chars, _batch_q_chars = [], [], [], []
	for i in xrange(hps.batch_size):
		padded_c_word = batch_c_word[i] + [0] * (max_c_length - batch_c_lens[i])
		padded_q_word = batch_q_word[i] + [0] * (max_q_length - batch_q_lens[i])
		padded_c_chars = batch_c_chars[i] + [[]] * (max_c_length - batch_c_lens[i])
		padded_q_chars = batch_q_chars[i] + [[]] * (max_q_length - batch_q_lens[i])
		_padded_c_chars, _padded_q_chars = [], []
		for c_c in padded_c_chars:
			padded_c_c = c_c + [0] * (max_char_length - len(c_c))
			_padded_c_chars.append(padded_c_c)
		for q_c in padded_q_chars:
			padded_q_c = q_c + [0] * (max_char_length - len(q_c))
			_padded_q_chars.append(padded_q_c)
		assert len(padded_c_word) == len(_padded_c_chars)
		assert len(padded_q_word) == len(_padded_q_chars)
		_batch_c_word.append(padded_c_word)
		_batch_q_word.append(padded_q_word)
		_batch_c_chars.append(_padded_c_chars)
		_batch_q_chars.append(_padded_q_chars)

	_batch_target = []
	for length_idx in xrange(2):
		_batch_target.append(
			np.array([batch_target[batch_idx][length_idx] for batch_idx in xrange(hps.batch_size)], dtype=np.int32))
	batch_c_chars = np.reshape(np.asarray(_batch_c_chars, dtype=np.int32),
							   newshape=(hps.batch_size * max_c_length, max_char_length))
	batch_q_chars = np.reshape(np.asarray(_batch_q_chars, dtype=np.int32),
							   newshape=(hps.batch_size * max_q_length, max_char_length))
	return (data, np.asarray(_batch_c_word, dtype=np.int32),
			np.asarray(_batch_q_word, dtype=np.int32),
			batch_c_chars, batch_q_chars, _batch_target,
			np.array(batch_c_lens, dtype=np.int32),
			np.array(batch_q_lens, dtype=np.int32),
			max_c_length, max_q_length, max_char_length)

def process_data():
	if not os.path.isfile(hps.data_dir + 'glove.np'):
		if os.path.exists(hps.train_dir):
			shutil.rmtree(hps.train_dir)
		if os.path.exists(hps.dev_dir):
			shutil.rmtree(hps.dev_dir)

		print("Reducing Glove Matrix")
		loader = data_loader(use_pretrained=False)
		loader.process_json(hps.data_dir + "train-v1.1.json", out_dir=hps.train_dir, write_=False)
		loader.process_json(hps.data_dir + "dev-v1.1.json", out_dir=hps.dev_dir, write_=False)
		reduce_glove(hps.glove_dir, loader.w_dict)
		with open(hps.data_dir + 'dictionary.pkl', 'wb') as dictionary:
			loader = data_loader(use_pretrained=True)
			print("Tokenizing training data.")
			loader.process_json(hps.data_dir + "train-v1.1.json", out_dir=hps.train_dir)
			print("Tokenizing dev data.")
			loader.process_json(hps.data_dir + "dev-v1.1.json", out_dir=hps.dev_dir)
			pickle.dump(loader, dictionary, pickle.HIGHEST_PROTOCOL)
		print("Tokenizing complete")
		load_glove(hps.glove_dir, "glove", vocab_size=loader.w_count)
		load_glove(hps.glove_char, "glove_char", vocab_size=loader.c_count)
		print("Processing complete")
		print("Unknown word ratio: {} / {}".format(loader.w_unknown_count, loader.w_occurence))
		print("Unknown character ratio: {} / {}".format(loader.c_unknown_count, loader.c_occurence))
	else:
		print("Processing complete")

if __name__ == '__main__':
	print()

