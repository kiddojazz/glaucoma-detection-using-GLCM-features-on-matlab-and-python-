{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Importing All Necessary Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn import preprocessing,neighbors\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import cross_validate\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import MinMaxScaler"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Giving the Glaucoma data a variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CONTRAST</th>\n",
       "      <th>CORRELATION</th>\n",
       "      <th>ENERGY</th>\n",
       "      <th>Homogeneity</th>\n",
       "      <th>ENTROPY</th>\n",
       "      <th>OUTPUT</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0944</td>\n",
       "      <td>0.9671</td>\n",
       "      <td>0.2584</td>\n",
       "      <td>0.9546</td>\n",
       "      <td>1.6474</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0817</td>\n",
       "      <td>0.9636</td>\n",
       "      <td>0.2808</td>\n",
       "      <td>0.9593</td>\n",
       "      <td>1.6774</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.0755</td>\n",
       "      <td>0.9685</td>\n",
       "      <td>0.2499</td>\n",
       "      <td>0.9628</td>\n",
       "      <td>1.6880</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.0811</td>\n",
       "      <td>0.9684</td>\n",
       "      <td>0.2312</td>\n",
       "      <td>0.9598</td>\n",
       "      <td>1.7332</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0520</td>\n",
       "      <td>0.9772</td>\n",
       "      <td>0.2860</td>\n",
       "      <td>0.9744</td>\n",
       "      <td>1.4939</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   CONTRAST  CORRELATION  ENERGY  Homogeneity  ENTROPY  OUTPUT\n",
       "0    0.0944       0.9671  0.2584       0.9546   1.6474       1\n",
       "1    0.0817       0.9636  0.2808       0.9593   1.6774       1\n",
       "2    0.0755       0.9685  0.2499       0.9628   1.6880       1\n",
       "3    0.0811       0.9684  0.2312       0.9598   1.7332       1\n",
       "4    0.0520       0.9772  0.2860       0.9744   1.4939       1"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train = pd.read_csv('Glaucoma.csv')\n",
    "train.replace('?', -99999, inplace=True)\n",
    "train.head() # the first 5 values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CONTRAST</th>\n",
       "      <th>CORRELATION</th>\n",
       "      <th>ENERGY</th>\n",
       "      <th>Homogeneity</th>\n",
       "      <th>ENTROPY</th>\n",
       "      <th>OUTPUT</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>85</th>\n",
       "      <td>0.0776</td>\n",
       "      <td>0.9412</td>\n",
       "      <td>0.3541</td>\n",
       "      <td>0.9614</td>\n",
       "      <td>1.4336</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86</th>\n",
       "      <td>0.0814</td>\n",
       "      <td>0.9442</td>\n",
       "      <td>0.3334</td>\n",
       "      <td>0.9595</td>\n",
       "      <td>1.4756</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>87</th>\n",
       "      <td>0.0857</td>\n",
       "      <td>0.9479</td>\n",
       "      <td>0.2766</td>\n",
       "      <td>0.9578</td>\n",
       "      <td>1.5995</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>88</th>\n",
       "      <td>0.0648</td>\n",
       "      <td>0.9618</td>\n",
       "      <td>0.2688</td>\n",
       "      <td>0.9676</td>\n",
       "      <td>1.5612</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>89</th>\n",
       "      <td>0.0590</td>\n",
       "      <td>0.9589</td>\n",
       "      <td>0.3579</td>\n",
       "      <td>0.9709</td>\n",
       "      <td>1.3845</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    CONTRAST  CORRELATION  ENERGY  Homogeneity  ENTROPY  OUTPUT\n",
       "85    0.0776       0.9412  0.3541       0.9614   1.4336       0\n",
       "86    0.0814       0.9442  0.3334       0.9595   1.4756       0\n",
       "87    0.0857       0.9479  0.2766       0.9578   1.5995       0\n",
       "88    0.0648       0.9618  0.2688       0.9676   1.5612       0\n",
       "89    0.0590       0.9589  0.3579       0.9709   1.3845       0"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.tail() # the last 5 values"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Creating Input (X) and Output(y) value of the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = np.array(train.drop('OUTPUT', axis=1))\n",
    "y = np.array(train['OUTPUT'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((90, 5), (90,))"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.shape, y.shape # Getting the shape of the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 90 entries, 0 to 89\n",
      "Data columns (total 6 columns):\n",
      "CONTRAST       90 non-null float64\n",
      "CORRELATION    90 non-null float64\n",
      "ENERGY         90 non-null float64\n",
      "Homogeneity    90 non-null float64\n",
      "ENTROPY        90 non-null float64\n",
      "OUTPUT         90 non-null int64\n",
      "dtypes: float64(5), int64(1)\n",
      "memory usage: 4.3 KB\n"
     ]
    }
   ],
   "source": [
    "train.info() # getting necessary info of type of data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CONTRAST</th>\n",
       "      <th>CORRELATION</th>\n",
       "      <th>ENERGY</th>\n",
       "      <th>Homogeneity</th>\n",
       "      <th>ENTROPY</th>\n",
       "      <th>OUTPUT</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>90.000000</td>\n",
       "      <td>90.000000</td>\n",
       "      <td>90.000000</td>\n",
       "      <td>90.000000</td>\n",
       "      <td>90.000000</td>\n",
       "      <td>90.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.080328</td>\n",
       "      <td>0.943364</td>\n",
       "      <td>0.346721</td>\n",
       "      <td>0.960652</td>\n",
       "      <td>1.399462</td>\n",
       "      <td>0.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.025021</td>\n",
       "      <td>0.026395</td>\n",
       "      <td>0.067754</td>\n",
       "      <td>0.038545</td>\n",
       "      <td>0.179321</td>\n",
       "      <td>0.502801</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.038200</td>\n",
       "      <td>0.876000</td>\n",
       "      <td>0.231200</td>\n",
       "      <td>0.606000</td>\n",
       "      <td>1.071300</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.061300</td>\n",
       "      <td>0.922150</td>\n",
       "      <td>0.288625</td>\n",
       "      <td>0.958500</td>\n",
       "      <td>1.266800</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.074350</td>\n",
       "      <td>0.949600</td>\n",
       "      <td>0.343000</td>\n",
       "      <td>0.965700</td>\n",
       "      <td>1.390150</td>\n",
       "      <td>0.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>0.096375</td>\n",
       "      <td>0.963600</td>\n",
       "      <td>0.383000</td>\n",
       "      <td>0.969700</td>\n",
       "      <td>1.553775</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>0.144800</td>\n",
       "      <td>0.978900</td>\n",
       "      <td>0.528600</td>\n",
       "      <td>0.981500</td>\n",
       "      <td>1.740400</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        CONTRAST  CORRELATION     ENERGY  Homogeneity    ENTROPY     OUTPUT\n",
       "count  90.000000    90.000000  90.000000    90.000000  90.000000  90.000000\n",
       "mean    0.080328     0.943364   0.346721     0.960652   1.399462   0.500000\n",
       "std     0.025021     0.026395   0.067754     0.038545   0.179321   0.502801\n",
       "min     0.038200     0.876000   0.231200     0.606000   1.071300   0.000000\n",
       "25%     0.061300     0.922150   0.288625     0.958500   1.266800   0.000000\n",
       "50%     0.074350     0.949600   0.343000     0.965700   1.390150   0.500000\n",
       "75%     0.096375     0.963600   0.383000     0.969700   1.553775   1.000000\n",
       "max     0.144800     0.978900   0.528600     0.981500   1.740400   1.000000"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.describe() # Getting the description of the data set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.shape[1]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Split data into train and test set. with test set of 20%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state= 0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data scaling and standardization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Feature scaling\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "sc = StandardScaler()\n",
    "X_train = sc.fit_transform(X_train)\n",
    "X_test = sc.fit_transform(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Creatin the neural network using keras\n",
    "and tensorflow as backend or we can use Theano"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "import keras \n",
    "from keras import backend as k\n",
    "from keras.models import Sequential\n",
    "from keras.layers.core import Dense, Dropout\n",
    "from keras.optimizers import Adam\n",
    "from keras.metrics import categorical_crossentropy\n",
    "from keras.callbacks import EarlyStopping\n",
    "\n",
    "classifier = Sequential() # Initialising the ANN\n",
    "\n",
    "n_cols = X.shape[1] # the X.shape is the number of input that would be inside the network\n",
    "\n",
    "classifier.add(Dense(16, input_dim=n_cols, activation='relu')) # input layer requires input_dim param\n",
    "classifier.add(Dense(32, activation='relu'))\n",
    "#classifier.add(Dropout(0.5))\n",
    "classifier.add(Dense(64, activation='relu'))\n",
    "classifier.add(Dense(32, activation='relu'))\n",
    "classifier.add(Dense(16, activation='relu'))\n",
    "classifier.add(Dropout(.2))\n",
    "classifier.add(Dense(2, activation='sigmoid')) # sigmoid instead of relu for final probability between 0 and 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense_7 (Dense)              (None, 16)                96        \n",
      "_________________________________________________________________\n",
      "dense_8 (Dense)              (None, 32)                544       \n",
      "_________________________________________________________________\n",
      "dense_9 (Dense)              (None, 64)                2112      \n",
      "_________________________________________________________________\n",
      "dense_10 (Dense)             (None, 32)                2080      \n",
      "_________________________________________________________________\n",
      "dense_11 (Dense)             (None, 16)                528       \n",
      "_________________________________________________________________\n",
      "dropout_2 (Dropout)          (None, 16)                0         \n",
      "_________________________________________________________________\n",
      "dense_12 (Dense)             (None, 2)                 34        \n",
      "=================================================================\n",
      "Total params: 5,394\n",
      "Trainable params: 5,394\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "classifier.summary() # gives the summary of the data set"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# The learning rate was set at 0.0001, the loss(entropy) and matric(accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "classifier.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Early stopping usually know as drawback was set a 3 \n",
    "which means if the accuracy remains the same after 3 iteration the training would stop.\n",
    "\n",
    "The draw back was later remove as it didnt allow the model to train more"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [],
   "source": [
    "early_stopping_monitor = EarlyStopping(patience=3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Importing Libraries for Data Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "import seaborn as sns\n",
    "sns.set() # setting seaborn default for plots"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 90 samples, validate on 18 samples\n",
      "Epoch 1/200\n",
      " - 1s - loss: 0.6916 - acc: 0.5222 - val_loss: 0.6876 - val_acc: 0.6667\n",
      "Epoch 2/200\n",
      " - 0s - loss: 0.6935 - acc: 0.5111 - val_loss: 0.6878 - val_acc: 0.7222\n",
      "Epoch 3/200\n",
      " - 0s - loss: 0.6934 - acc: 0.4778 - val_loss: 0.6883 - val_acc: 0.6111\n",
      "Epoch 4/200\n",
      " - 0s - loss: 0.6918 - acc: 0.5444 - val_loss: 0.6884 - val_acc: 0.6111\n",
      "Epoch 5/200\n",
      " - 0s - loss: 0.6925 - acc: 0.5889 - val_loss: 0.6878 - val_acc: 0.7222\n",
      "Epoch 6/200\n",
      " - 0s - loss: 0.6925 - acc: 0.5000 - val_loss: 0.6885 - val_acc: 0.6111\n",
      "Epoch 7/200\n",
      " - 0s - loss: 0.6936 - acc: 0.5000 - val_loss: 0.6896 - val_acc: 0.5556\n",
      "Epoch 8/200\n",
      " - 0s - loss: 0.6916 - acc: 0.5111 - val_loss: 0.6892 - val_acc: 0.5556\n",
      "Epoch 9/200\n",
      " - 0s - loss: 0.6932 - acc: 0.5667 - val_loss: 0.6880 - val_acc: 0.6667\n",
      "Epoch 10/200\n",
      " - 0s - loss: 0.6915 - acc: 0.5778 - val_loss: 0.6885 - val_acc: 0.6667\n",
      "Epoch 11/200\n",
      " - 0s - loss: 0.6930 - acc: 0.5111 - val_loss: 0.6878 - val_acc: 0.7222\n",
      "Epoch 12/200\n",
      " - 0s - loss: 0.6923 - acc: 0.5333 - val_loss: 0.6881 - val_acc: 0.6667\n",
      "Epoch 13/200\n",
      " - 0s - loss: 0.6941 - acc: 0.4667 - val_loss: 0.6887 - val_acc: 0.6667\n",
      "Epoch 14/200\n",
      " - 0s - loss: 0.6908 - acc: 0.5667 - val_loss: 0.6895 - val_acc: 0.5556\n",
      "Epoch 15/200\n",
      " - 0s - loss: 0.6916 - acc: 0.5667 - val_loss: 0.6888 - val_acc: 0.6667\n",
      "Epoch 16/200\n",
      " - 0s - loss: 0.6906 - acc: 0.5667 - val_loss: 0.6883 - val_acc: 0.5000\n",
      "Epoch 17/200\n",
      " - 0s - loss: 0.6912 - acc: 0.5667 - val_loss: 0.6878 - val_acc: 0.6111\n",
      "Epoch 18/200\n",
      " - 0s - loss: 0.6912 - acc: 0.5556 - val_loss: 0.6878 - val_acc: 0.5556\n",
      "Epoch 19/200\n",
      " - 0s - loss: 0.6932 - acc: 0.5111 - val_loss: 0.6882 - val_acc: 0.5556\n",
      "Epoch 20/200\n",
      " - 0s - loss: 0.6911 - acc: 0.5333 - val_loss: 0.6878 - val_acc: 0.5556\n",
      "Epoch 21/200\n",
      " - 0s - loss: 0.6924 - acc: 0.5222 - val_loss: 0.6889 - val_acc: 0.5556\n",
      "Epoch 22/200\n",
      " - 0s - loss: 0.6914 - acc: 0.5333 - val_loss: 0.6882 - val_acc: 0.5556\n",
      "Epoch 23/200\n",
      " - 0s - loss: 0.6911 - acc: 0.5556 - val_loss: 0.6884 - val_acc: 0.5000\n",
      "Epoch 24/200\n",
      " - 0s - loss: 0.6884 - acc: 0.5444 - val_loss: 0.6873 - val_acc: 0.5556\n",
      "Epoch 25/200\n",
      " - 0s - loss: 0.6900 - acc: 0.5556 - val_loss: 0.6872 - val_acc: 0.5556\n",
      "Epoch 26/200\n",
      " - 0s - loss: 0.6921 - acc: 0.5444 - val_loss: 0.6874 - val_acc: 0.5000\n",
      "Epoch 27/200\n",
      " - 0s - loss: 0.6928 - acc: 0.4778 - val_loss: 0.6876 - val_acc: 0.5000\n",
      "Epoch 28/200\n",
      " - 0s - loss: 0.6902 - acc: 0.5778 - val_loss: 0.6863 - val_acc: 0.6111\n",
      "Epoch 29/200\n",
      " - 0s - loss: 0.6896 - acc: 0.5778 - val_loss: 0.6866 - val_acc: 0.5556\n",
      "Epoch 30/200\n",
      " - 0s - loss: 0.6906 - acc: 0.5222 - val_loss: 0.6872 - val_acc: 0.5000\n",
      "Epoch 31/200\n",
      " - 0s - loss: 0.6896 - acc: 0.5444 - val_loss: 0.6879 - val_acc: 0.5000\n",
      "Epoch 32/200\n",
      " - 0s - loss: 0.6927 - acc: 0.5333 - val_loss: 0.6879 - val_acc: 0.5000\n",
      "Epoch 33/200\n",
      " - 0s - loss: 0.6907 - acc: 0.5444 - val_loss: 0.6866 - val_acc: 0.6111\n",
      "Epoch 34/200\n",
      " - 0s - loss: 0.6915 - acc: 0.5000 - val_loss: 0.6875 - val_acc: 0.5000\n",
      "Epoch 35/200\n",
      " - 0s - loss: 0.6901 - acc: 0.5444 - val_loss: 0.6862 - val_acc: 0.6111\n",
      "Epoch 36/200\n",
      " - 0s - loss: 0.6915 - acc: 0.4667 - val_loss: 0.6863 - val_acc: 0.6111\n",
      "Epoch 37/200\n",
      " - 0s - loss: 0.6913 - acc: 0.5111 - val_loss: 0.6858 - val_acc: 0.6111\n",
      "Epoch 38/200\n",
      " - 0s - loss: 0.6910 - acc: 0.5444 - val_loss: 0.6868 - val_acc: 0.5556\n",
      "Epoch 39/200\n",
      " - 0s - loss: 0.6907 - acc: 0.5000 - val_loss: 0.6867 - val_acc: 0.5556\n",
      "Epoch 40/200\n",
      " - 0s - loss: 0.6891 - acc: 0.5889 - val_loss: 0.6863 - val_acc: 0.6111\n",
      "Epoch 41/200\n",
      " - 0s - loss: 0.6894 - acc: 0.5667 - val_loss: 0.6864 - val_acc: 0.6111\n",
      "Epoch 42/200\n",
      " - 0s - loss: 0.6884 - acc: 0.5667 - val_loss: 0.6869 - val_acc: 0.5000\n",
      "Epoch 43/200\n",
      " - 0s - loss: 0.6911 - acc: 0.5222 - val_loss: 0.6875 - val_acc: 0.5000\n",
      "Epoch 44/200\n",
      " - 0s - loss: 0.6906 - acc: 0.4889 - val_loss: 0.6871 - val_acc: 0.5000\n",
      "Epoch 45/200\n",
      " - 0s - loss: 0.6889 - acc: 0.5778 - val_loss: 0.6864 - val_acc: 0.5000\n",
      "Epoch 46/200\n",
      " - 0s - loss: 0.6876 - acc: 0.5667 - val_loss: 0.6857 - val_acc: 0.6111\n",
      "Epoch 47/200\n",
      " - 0s - loss: 0.6873 - acc: 0.5667 - val_loss: 0.6857 - val_acc: 0.5556\n",
      "Epoch 48/200\n",
      " - 0s - loss: 0.6842 - acc: 0.6556 - val_loss: 0.6845 - val_acc: 0.5556\n",
      "Epoch 49/200\n",
      " - 0s - loss: 0.6887 - acc: 0.5778 - val_loss: 0.6838 - val_acc: 0.6111\n",
      "Epoch 50/200\n",
      " - 0s - loss: 0.6842 - acc: 0.6222 - val_loss: 0.6827 - val_acc: 0.6111\n",
      "Epoch 51/200\n",
      " - 0s - loss: 0.6904 - acc: 0.5000 - val_loss: 0.6830 - val_acc: 0.6111\n",
      "Epoch 52/200\n",
      " - 0s - loss: 0.6826 - acc: 0.6444 - val_loss: 0.6814 - val_acc: 0.6111\n",
      "Epoch 53/200\n",
      " - 0s - loss: 0.6779 - acc: 0.6667 - val_loss: 0.6795 - val_acc: 0.6111\n",
      "Epoch 54/200\n",
      " - 0s - loss: 0.6913 - acc: 0.5000 - val_loss: 0.6805 - val_acc: 0.6111\n",
      "Epoch 55/200\n",
      " - 0s - loss: 0.6893 - acc: 0.4778 - val_loss: 0.6814 - val_acc: 0.6111\n",
      "Epoch 56/200\n",
      " - 0s - loss: 0.6891 - acc: 0.4778 - val_loss: 0.6816 - val_acc: 0.6111\n",
      "Epoch 57/200\n",
      " - 0s - loss: 0.6871 - acc: 0.5000 - val_loss: 0.6822 - val_acc: 0.5556\n",
      "Epoch 58/200\n",
      " - 0s - loss: 0.6876 - acc: 0.5333 - val_loss: 0.6817 - val_acc: 0.6111\n",
      "Epoch 59/200\n",
      " - 0s - loss: 0.6845 - acc: 0.5889 - val_loss: 0.6816 - val_acc: 0.6111\n",
      "Epoch 60/200\n",
      " - 0s - loss: 0.6792 - acc: 0.6556 - val_loss: 0.6801 - val_acc: 0.6111\n",
      "Epoch 61/200\n",
      " - 0s - loss: 0.6812 - acc: 0.6000 - val_loss: 0.6797 - val_acc: 0.6111\n",
      "Epoch 62/200\n",
      " - 0s - loss: 0.6847 - acc: 0.5556 - val_loss: 0.6795 - val_acc: 0.5556\n",
      "Epoch 63/200\n",
      " - 0s - loss: 0.6865 - acc: 0.5444 - val_loss: 0.6797 - val_acc: 0.5556\n",
      "Epoch 64/200\n",
      " - 0s - loss: 0.6820 - acc: 0.6222 - val_loss: 0.6795 - val_acc: 0.5556\n",
      "Epoch 65/200\n",
      " - 0s - loss: 0.6814 - acc: 0.6444 - val_loss: 0.6779 - val_acc: 0.6111\n",
      "Epoch 66/200\n",
      " - 0s - loss: 0.6848 - acc: 0.5444 - val_loss: 0.6774 - val_acc: 0.5556\n",
      "Epoch 67/200\n",
      " - 0s - loss: 0.6833 - acc: 0.5667 - val_loss: 0.6768 - val_acc: 0.6111\n",
      "Epoch 68/200\n",
      " - 0s - loss: 0.6866 - acc: 0.5000 - val_loss: 0.6767 - val_acc: 0.6111\n",
      "Epoch 69/200\n",
      " - 0s - loss: 0.6812 - acc: 0.6111 - val_loss: 0.6764 - val_acc: 0.6111\n",
      "Epoch 70/200\n",
      " - 0s - loss: 0.6899 - acc: 0.5556 - val_loss: 0.6764 - val_acc: 0.6111\n",
      "Epoch 71/200\n",
      " - 0s - loss: 0.6854 - acc: 0.5444 - val_loss: 0.6754 - val_acc: 0.6111\n",
      "Epoch 72/200\n",
      " - 0s - loss: 0.6800 - acc: 0.6556 - val_loss: 0.6746 - val_acc: 0.6111\n",
      "Epoch 73/200\n",
      " - 0s - loss: 0.6810 - acc: 0.6000 - val_loss: 0.6737 - val_acc: 0.6111\n",
      "Epoch 74/200\n",
      " - 0s - loss: 0.6804 - acc: 0.5778 - val_loss: 0.6721 - val_acc: 0.6111\n",
      "Epoch 75/200\n",
      " - 0s - loss: 0.6843 - acc: 0.5333 - val_loss: 0.6716 - val_acc: 0.6111\n",
      "Epoch 76/200\n",
      " - 0s - loss: 0.6801 - acc: 0.6111 - val_loss: 0.6701 - val_acc: 0.6111\n",
      "Epoch 77/200\n",
      " - 0s - loss: 0.6695 - acc: 0.7111 - val_loss: 0.6683 - val_acc: 0.6111\n",
      "Epoch 78/200\n",
      " - 0s - loss: 0.6768 - acc: 0.6556 - val_loss: 0.6684 - val_acc: 0.6111\n",
      "Epoch 79/200\n",
      " - 0s - loss: 0.6744 - acc: 0.5889 - val_loss: 0.6677 - val_acc: 0.6111\n",
      "Epoch 80/200\n",
      " - 0s - loss: 0.6875 - acc: 0.4889 - val_loss: 0.6673 - val_acc: 0.6111\n",
      "Epoch 81/200\n",
      " - 0s - loss: 0.6760 - acc: 0.5778 - val_loss: 0.6677 - val_acc: 0.6111\n",
      "Epoch 82/200\n",
      " - 0s - loss: 0.6786 - acc: 0.5778 - val_loss: 0.6697 - val_acc: 0.6111\n",
      "Epoch 83/200\n",
      " - 0s - loss: 0.6723 - acc: 0.6000 - val_loss: 0.6689 - val_acc: 0.6111\n",
      "Epoch 84/200\n",
      " - 0s - loss: 0.6741 - acc: 0.6222 - val_loss: 0.6673 - val_acc: 0.6111\n",
      "Epoch 85/200\n",
      " - 0s - loss: 0.6803 - acc: 0.5778 - val_loss: 0.6650 - val_acc: 0.6111\n",
      "Epoch 86/200\n",
      " - 0s - loss: 0.6761 - acc: 0.6222 - val_loss: 0.6667 - val_acc: 0.6111\n",
      "Epoch 87/200\n",
      " - 0s - loss: 0.6696 - acc: 0.6000 - val_loss: 0.6675 - val_acc: 0.6111\n",
      "Epoch 88/200\n",
      " - 0s - loss: 0.6687 - acc: 0.6444 - val_loss: 0.6657 - val_acc: 0.6111\n",
      "Epoch 89/200\n",
      " - 0s - loss: 0.6736 - acc: 0.6000 - val_loss: 0.6652 - val_acc: 0.6111\n",
      "Epoch 90/200\n",
      " - 0s - loss: 0.6653 - acc: 0.6667 - val_loss: 0.6645 - val_acc: 0.6111\n",
      "Epoch 91/200\n",
      " - 0s - loss: 0.6678 - acc: 0.6556 - val_loss: 0.6653 - val_acc: 0.6111\n",
      "Epoch 92/200\n",
      " - 0s - loss: 0.6627 - acc: 0.6778 - val_loss: 0.6644 - val_acc: 0.6111\n",
      "Epoch 93/200\n",
      " - 0s - loss: 0.6588 - acc: 0.6889 - val_loss: 0.6637 - val_acc: 0.6111\n",
      "Epoch 94/200\n",
      " - 0s - loss: 0.6729 - acc: 0.5889 - val_loss: 0.6633 - val_acc: 0.6111\n",
      "Epoch 95/200\n",
      " - 0s - loss: 0.6615 - acc: 0.6667 - val_loss: 0.6641 - val_acc: 0.6111\n",
      "Epoch 96/200\n",
      " - 0s - loss: 0.6521 - acc: 0.6889 - val_loss: 0.6643 - val_acc: 0.6111\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 97/200\n",
      " - 0s - loss: 0.6616 - acc: 0.5889 - val_loss: 0.6595 - val_acc: 0.6111\n",
      "Epoch 98/200\n",
      " - 0s - loss: 0.6565 - acc: 0.6556 - val_loss: 0.6630 - val_acc: 0.6111\n",
      "Epoch 99/200\n",
      " - 0s - loss: 0.6642 - acc: 0.6333 - val_loss: 0.6611 - val_acc: 0.6111\n",
      "Epoch 100/200\n",
      " - 0s - loss: 0.6487 - acc: 0.6667 - val_loss: 0.6636 - val_acc: 0.6111\n",
      "Epoch 101/200\n",
      " - 0s - loss: 0.6518 - acc: 0.6889 - val_loss: 0.6629 - val_acc: 0.6111\n",
      "Epoch 102/200\n",
      " - 0s - loss: 0.6610 - acc: 0.6000 - val_loss: 0.6631 - val_acc: 0.6111\n",
      "Epoch 103/200\n",
      " - 0s - loss: 0.6582 - acc: 0.6333 - val_loss: 0.6629 - val_acc: 0.6111\n",
      "Epoch 104/200\n",
      " - 0s - loss: 0.6497 - acc: 0.6556 - val_loss: 0.6662 - val_acc: 0.6111\n",
      "Epoch 105/200\n",
      " - 0s - loss: 0.6467 - acc: 0.6444 - val_loss: 0.6660 - val_acc: 0.6111\n",
      "Epoch 106/200\n",
      " - 0s - loss: 0.6469 - acc: 0.6444 - val_loss: 0.6618 - val_acc: 0.6111\n",
      "Epoch 107/200\n",
      " - 0s - loss: 0.6427 - acc: 0.7222 - val_loss: 0.6681 - val_acc: 0.6111\n",
      "Epoch 108/200\n",
      " - 0s - loss: 0.6430 - acc: 0.6333 - val_loss: 0.6732 - val_acc: 0.6111\n",
      "Epoch 109/200\n",
      " - 0s - loss: 0.6418 - acc: 0.6667 - val_loss: 0.6707 - val_acc: 0.6111\n",
      "Epoch 110/200\n",
      " - 0s - loss: 0.6425 - acc: 0.6556 - val_loss: 0.6718 - val_acc: 0.6111\n",
      "Epoch 111/200\n",
      " - 0s - loss: 0.6488 - acc: 0.6889 - val_loss: 0.6735 - val_acc: 0.6111\n",
      "Epoch 112/200\n",
      " - 0s - loss: 0.6373 - acc: 0.6667 - val_loss: 0.6766 - val_acc: 0.6111\n",
      "Epoch 113/200\n",
      " - 0s - loss: 0.6340 - acc: 0.6889 - val_loss: 0.6751 - val_acc: 0.6111\n",
      "Epoch 114/200\n",
      " - 0s - loss: 0.6467 - acc: 0.6667 - val_loss: 0.6799 - val_acc: 0.6111\n",
      "Epoch 115/200\n",
      " - 0s - loss: 0.6418 - acc: 0.6778 - val_loss: 0.6812 - val_acc: 0.6111\n",
      "Epoch 116/200\n",
      " - 0s - loss: 0.6411 - acc: 0.6778 - val_loss: 0.6782 - val_acc: 0.6111\n",
      "Epoch 117/200\n",
      " - 0s - loss: 0.6382 - acc: 0.7000 - val_loss: 0.6858 - val_acc: 0.6111\n",
      "Epoch 118/200\n",
      " - 0s - loss: 0.6270 - acc: 0.7111 - val_loss: 0.6931 - val_acc: 0.6111\n",
      "Epoch 119/200\n",
      " - 0s - loss: 0.6376 - acc: 0.6667 - val_loss: 0.6905 - val_acc: 0.6111\n",
      "Epoch 120/200\n",
      " - 0s - loss: 0.6305 - acc: 0.6889 - val_loss: 0.6946 - val_acc: 0.6111\n",
      "Epoch 121/200\n",
      " - 0s - loss: 0.6234 - acc: 0.7111 - val_loss: 0.6989 - val_acc: 0.6111\n",
      "Epoch 122/200\n",
      " - 0s - loss: 0.6112 - acc: 0.7111 - val_loss: 0.7043 - val_acc: 0.6111\n",
      "Epoch 123/200\n",
      " - 0s - loss: 0.6353 - acc: 0.6778 - val_loss: 0.7105 - val_acc: 0.6111\n",
      "Epoch 124/200\n",
      " - 0s - loss: 0.6155 - acc: 0.7556 - val_loss: 0.7054 - val_acc: 0.6111\n",
      "Epoch 125/200\n",
      " - 0s - loss: 0.6060 - acc: 0.7667 - val_loss: 0.7263 - val_acc: 0.5556\n",
      "Epoch 126/200\n",
      " - 0s - loss: 0.6194 - acc: 0.7556 - val_loss: 0.7152 - val_acc: 0.6111\n",
      "Epoch 127/200\n",
      " - 0s - loss: 0.6030 - acc: 0.7222 - val_loss: 0.7341 - val_acc: 0.5556\n",
      "Epoch 128/200\n",
      " - 0s - loss: 0.6090 - acc: 0.7000 - val_loss: 0.7284 - val_acc: 0.6111\n",
      "Epoch 129/200\n",
      " - 0s - loss: 0.6103 - acc: 0.7222 - val_loss: 0.7373 - val_acc: 0.6111\n",
      "Epoch 130/200\n",
      " - 0s - loss: 0.6100 - acc: 0.6778 - val_loss: 0.7422 - val_acc: 0.6111\n",
      "Epoch 131/200\n",
      " - 0s - loss: 0.6241 - acc: 0.6778 - val_loss: 0.7462 - val_acc: 0.6111\n",
      "Epoch 132/200\n",
      " - 0s - loss: 0.5867 - acc: 0.8000 - val_loss: 0.7520 - val_acc: 0.6111\n",
      "Epoch 133/200\n",
      " - 0s - loss: 0.6133 - acc: 0.7333 - val_loss: 0.7470 - val_acc: 0.6111\n",
      "Epoch 134/200\n",
      " - 0s - loss: 0.6049 - acc: 0.7333 - val_loss: 0.7540 - val_acc: 0.6111\n",
      "Epoch 135/200\n",
      " - 0s - loss: 0.6025 - acc: 0.7111 - val_loss: 0.7729 - val_acc: 0.6111\n",
      "Epoch 136/200\n",
      " - 0s - loss: 0.5960 - acc: 0.7333 - val_loss: 0.7882 - val_acc: 0.5556\n",
      "Epoch 137/200\n",
      " - 0s - loss: 0.5801 - acc: 0.7444 - val_loss: 0.7715 - val_acc: 0.6111\n",
      "Epoch 138/200\n",
      " - 0s - loss: 0.5862 - acc: 0.7444 - val_loss: 0.7903 - val_acc: 0.6111\n",
      "Epoch 139/200\n",
      " - 0s - loss: 0.6186 - acc: 0.6778 - val_loss: 0.7997 - val_acc: 0.6111\n",
      "Epoch 140/200\n",
      " - 0s - loss: 0.5928 - acc: 0.7000 - val_loss: 0.8075 - val_acc: 0.6111\n",
      "Epoch 141/200\n",
      " - 0s - loss: 0.5889 - acc: 0.7333 - val_loss: 0.8143 - val_acc: 0.6111\n",
      "Epoch 142/200\n",
      " - 0s - loss: 0.5764 - acc: 0.7333 - val_loss: 0.8078 - val_acc: 0.6111\n",
      "Epoch 143/200\n",
      " - 0s - loss: 0.5950 - acc: 0.6889 - val_loss: 0.8231 - val_acc: 0.6111\n",
      "Epoch 144/200\n",
      " - 0s - loss: 0.5891 - acc: 0.7778 - val_loss: 0.8366 - val_acc: 0.6111\n",
      "Epoch 145/200\n",
      " - 0s - loss: 0.5832 - acc: 0.7222 - val_loss: 0.8383 - val_acc: 0.6111\n",
      "Epoch 146/200\n",
      " - 0s - loss: 0.5888 - acc: 0.7333 - val_loss: 0.8322 - val_acc: 0.6111\n",
      "Epoch 147/200\n",
      " - 0s - loss: 0.5934 - acc: 0.7222 - val_loss: 0.8546 - val_acc: 0.6111\n",
      "Epoch 148/200\n",
      " - 0s - loss: 0.5846 - acc: 0.7222 - val_loss: 0.8589 - val_acc: 0.6111\n",
      "Epoch 149/200\n",
      " - 0s - loss: 0.5751 - acc: 0.6778 - val_loss: 0.8553 - val_acc: 0.6111\n",
      "Epoch 150/200\n",
      " - 0s - loss: 0.5840 - acc: 0.7111 - val_loss: 0.8712 - val_acc: 0.6111\n",
      "Epoch 151/200\n",
      " - 0s - loss: 0.5566 - acc: 0.7667 - val_loss: 0.8865 - val_acc: 0.6111\n",
      "Epoch 152/200\n",
      " - 0s - loss: 0.5857 - acc: 0.6778 - val_loss: 0.8875 - val_acc: 0.6111\n",
      "Epoch 153/200\n",
      " - 0s - loss: 0.5871 - acc: 0.6778 - val_loss: 0.9075 - val_acc: 0.6111\n",
      "Epoch 154/200\n",
      " - 0s - loss: 0.5785 - acc: 0.7333 - val_loss: 0.9233 - val_acc: 0.6111\n",
      "Epoch 155/200\n",
      " - 0s - loss: 0.5497 - acc: 0.7333 - val_loss: 0.9242 - val_acc: 0.6111\n",
      "Epoch 156/200\n",
      " - 0s - loss: 0.5628 - acc: 0.6778 - val_loss: 0.9244 - val_acc: 0.6111\n",
      "Epoch 157/200\n",
      " - 0s - loss: 0.5381 - acc: 0.7556 - val_loss: 0.9266 - val_acc: 0.6111\n",
      "Epoch 158/200\n",
      " - 0s - loss: 0.5729 - acc: 0.7000 - val_loss: 0.9543 - val_acc: 0.6111\n",
      "Epoch 159/200\n",
      " - 0s - loss: 0.5525 - acc: 0.7444 - val_loss: 0.9620 - val_acc: 0.6111\n",
      "Epoch 160/200\n",
      " - 0s - loss: 0.5399 - acc: 0.7444 - val_loss: 0.9869 - val_acc: 0.6111\n",
      "Epoch 161/200\n",
      " - 0s - loss: 0.5465 - acc: 0.7333 - val_loss: 0.9731 - val_acc: 0.6111\n",
      "Epoch 162/200\n",
      " - 0s - loss: 0.5424 - acc: 0.7556 - val_loss: 1.0030 - val_acc: 0.6111\n",
      "Epoch 163/200\n",
      " - 0s - loss: 0.5388 - acc: 0.7667 - val_loss: 1.0170 - val_acc: 0.6111\n",
      "Epoch 164/200\n",
      " - 0s - loss: 0.5589 - acc: 0.7444 - val_loss: 1.0334 - val_acc: 0.6111\n",
      "Epoch 165/200\n",
      " - 0s - loss: 0.5492 - acc: 0.7333 - val_loss: 1.0346 - val_acc: 0.6111\n",
      "Epoch 166/200\n",
      " - 0s - loss: 0.5426 - acc: 0.7556 - val_loss: 1.0545 - val_acc: 0.6111\n",
      "Epoch 167/200\n",
      " - 0s - loss: 0.5365 - acc: 0.7333 - val_loss: 1.0584 - val_acc: 0.6111\n",
      "Epoch 168/200\n",
      " - 0s - loss: 0.5408 - acc: 0.6889 - val_loss: 1.0527 - val_acc: 0.6111\n",
      "Epoch 169/200\n",
      " - 0s - loss: 0.5397 - acc: 0.7444 - val_loss: 1.0517 - val_acc: 0.6111\n",
      "Epoch 170/200\n",
      " - 0s - loss: 0.5363 - acc: 0.6667 - val_loss: 1.0687 - val_acc: 0.6111\n",
      "Epoch 171/200\n",
      " - 0s - loss: 0.5146 - acc: 0.8000 - val_loss: 1.0972 - val_acc: 0.6111\n",
      "Epoch 172/200\n",
      " - 0s - loss: 0.5292 - acc: 0.7444 - val_loss: 1.1119 - val_acc: 0.6111\n",
      "Epoch 173/200\n",
      " - 0s - loss: 0.5372 - acc: 0.7222 - val_loss: 1.1311 - val_acc: 0.6111\n",
      "Epoch 174/200\n",
      " - 0s - loss: 0.5258 - acc: 0.7778 - val_loss: 1.1086 - val_acc: 0.6111\n",
      "Epoch 175/200\n",
      " - 0s - loss: 0.5567 - acc: 0.7667 - val_loss: 1.1018 - val_acc: 0.6111\n",
      "Epoch 176/200\n",
      " - 0s - loss: 0.5382 - acc: 0.7111 - val_loss: 1.1059 - val_acc: 0.6111\n",
      "Epoch 177/200\n",
      " - 0s - loss: 0.5482 - acc: 0.7778 - val_loss: 1.1584 - val_acc: 0.6111\n",
      "Epoch 178/200\n",
      " - 0s - loss: 0.5175 - acc: 0.7556 - val_loss: 1.1448 - val_acc: 0.6111\n",
      "Epoch 179/200\n",
      " - 0s - loss: 0.5335 - acc: 0.7444 - val_loss: 1.1705 - val_acc: 0.6111\n",
      "Epoch 180/200\n",
      " - 0s - loss: 0.5231 - acc: 0.7222 - val_loss: 1.1711 - val_acc: 0.6111\n",
      "Epoch 181/200\n",
      " - 0s - loss: 0.5249 - acc: 0.7667 - val_loss: 1.1797 - val_acc: 0.6111\n",
      "Epoch 182/200\n",
      " - 0s - loss: 0.5400 - acc: 0.7000 - val_loss: 1.2051 - val_acc: 0.6111\n",
      "Epoch 183/200\n",
      " - 0s - loss: 0.5024 - acc: 0.8000 - val_loss: 1.2001 - val_acc: 0.6111\n",
      "Epoch 184/200\n",
      " - 0s - loss: 0.5009 - acc: 0.7778 - val_loss: 1.2217 - val_acc: 0.6111\n",
      "Epoch 185/200\n",
      " - 0s - loss: 0.5351 - acc: 0.7444 - val_loss: 1.2308 - val_acc: 0.6111\n",
      "Epoch 186/200\n",
      " - 0s - loss: 0.5208 - acc: 0.7222 - val_loss: 1.2264 - val_acc: 0.6111\n",
      "Epoch 187/200\n",
      " - 0s - loss: 0.5170 - acc: 0.7556 - val_loss: 1.2446 - val_acc: 0.6111\n",
      "Epoch 188/200\n",
      " - 0s - loss: 0.4992 - acc: 0.7889 - val_loss: 1.2783 - val_acc: 0.6111\n",
      "Epoch 189/200\n",
      " - 0s - loss: 0.5100 - acc: 0.7444 - val_loss: 1.2683 - val_acc: 0.6111\n",
      "Epoch 190/200\n",
      " - 0s - loss: 0.5214 - acc: 0.7333 - val_loss: 1.2959 - val_acc: 0.6111\n",
      "Epoch 191/200\n",
      " - 0s - loss: 0.4668 - acc: 0.8000 - val_loss: 1.2812 - val_acc: 0.6111\n",
      "Epoch 192/200\n",
      " - 0s - loss: 0.5297 - acc: 0.8000 - val_loss: 1.2937 - val_acc: 0.6111\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 193/200\n",
      " - 0s - loss: 0.5153 - acc: 0.7444 - val_loss: 1.3399 - val_acc: 0.6111\n",
      "Epoch 194/200\n",
      " - 0s - loss: 0.5181 - acc: 0.6889 - val_loss: 1.2907 - val_acc: 0.6111\n",
      "Epoch 195/200\n",
      " - 0s - loss: 0.5007 - acc: 0.7889 - val_loss: 1.3202 - val_acc: 0.6111\n",
      "Epoch 196/200\n",
      " - 0s - loss: 0.5411 - acc: 0.7556 - val_loss: 1.3271 - val_acc: 0.6111\n",
      "Epoch 197/200\n",
      " - 0s - loss: 0.5198 - acc: 0.7333 - val_loss: 1.3328 - val_acc: 0.6111\n",
      "Epoch 198/200\n",
      " - 0s - loss: 0.4799 - acc: 0.7889 - val_loss: 1.3545 - val_acc: 0.6111\n",
      "Epoch 199/200\n",
      " - 0s - loss: 0.5243 - acc: 0.7444 - val_loss: 1.3652 - val_acc: 0.6111\n",
      "Epoch 200/200\n",
      " - 0s - loss: 0.5189 - acc: 0.7333 - val_loss: 1.3805 - val_acc: 0.6111\n"
     ]
    }
   ],
   "source": [
    "history = classifier.fit(X, y, validation_split=0.2, batch_size = 1, epochs = 200, shuffle=True, verbose=2,  validation_data=(X_test, y_test)) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n"
     ]
    }
   ],
   "source": [
    "Y_pred = classifier.predict(X_test, batch_size=10, verbose=0)\n",
    "#Y_pred = (Y_pred>0.5)\n",
    "#Y_pred = [ 1 if y>=0.5 else 0 for y in Y_pred ]\n",
    "print(np.argmax(Y_pred[0])) # first test for prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "72/72 [==============================] - 0s 69us/step\n",
      "acc: 73.61%\n"
     ]
    }
   ],
   "source": [
    "scores = classifier.evaluate(X_train, y_train, batch_size=30)\n",
    "print(\"%s: %.2f%%\" % (classifier.metrics_names[1], scores[1]*100)) # evaluation score was found to be 77.78%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZAAAAEXCAYAAACDChKsAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzsvXmAVNWdPX7eVltX9VZUNyiIIoiOgqIoBteoCVFoJS6jE3ejX0fjGM1IgksSNWMWFcU4btFEZ0JccJdo+BnjGKNgFGMUEJV9p5eq7q6qru0t9/fHe/fWe7X3Xk3f84/dVW+5VTb3vPM5n0UghBBwcHBwcHD0EuJwL4CDg4ODY2SCEwgHBwcHR5/ACYSDg4ODo0/gBMLBwcHB0SdwAuHg4ODg6BM4gXBwcHBw9AmcQDhGPHbs2IGpU6fioosuyntv4cKFmDp1KiKRSK+uefXVV+Oll14qeczf//53zJs3r+j7qqri+OOPx5VXXtmre3NwjBRwAuHYK+B2u7F582bs3LmTvZZIJPCPf/xj2Nb05z//GQcffDDWrFmDjRs3Dts6ODgGC5xAOPYKSJKE008/HcuWLWOvvfnmmzj11FMdxz333HOYN28ezjzzTFxxxRXYvHkzAKC1tRWXX3455s6di6uuugrt7e3snI0bN+KKK67A2WefjbPOOgsvvPBCRWt65plncOqpp+KMM87A//zP/zjee+GFFzB37ly0tLTgkksuwe7du4u+nqt07L8/+OCD+O53v4uWlhbcdNNN6OjowLXXXovzzz8fp5xyCi6++GKEw2EAwObNm3HxxRez67/xxhv4+OOPcfLJJ8MwDABAMpnE1772tV4rNo5RCsLBMcKxfft2csQRR5DVq1eTb33rW+z1Sy+9lHz55ZfkoIMOIuFwmKxYsYKcdtppJBwOE0IIefHFF8npp59ODMMg1157Lbn//vsJIYRs2bKFHHHEEeTFF18kqqqSM844g6xZs4YQQkg0GiWnn346+eSTT8gHH3xA5s6dW3BN69evJ4ceeiiJRCLk008/JdOnTyeRSIQQQsi6devIrFmzyK5duwghhDz55JPkxz/+cdHXc+9j//3Xv/41mTNnDlFVlRBCyFNPPUUee+wxQgghhmGQK6+8kvz2t78lhBAyf/58smTJEkIIIbt27SKnnnoqicVi5MwzzyTvvPMOIYSQ559/ntx44439+v/BMXogDzeBcXAMFA477DBIkoQ1a9YgGAyip6cHBx10EHv/b3/7G8444ww0NjYCAM4++2zcdddd2LFjB1asWIEf/ehHAICJEydi1qxZAIAtW7Zg27ZtuOWWW9h1UqkUPv/8cxx44IFF1/LMM8/g61//OhoaGtDQ0IDx48dj6dKluPrqq7Fy5Uocf/zxGDduHADgsssuAwA8+eSTBV//+9//XvJzH3HEEZBl85/ypZdeilWrVuHJJ5/Eli1bsH79ehx++OHo6urCF198gfPOOw8AMG7cOLz11lsAgAsvvBBLly7FSSedhOeeew4//OEPy3/ZHBwAOIFw7FU488wz8dprr6GxsRFnnXWW4z0aprGDEAJN0yAIAoitLRzdkHVdRyAQwKuvvsre6+joQCAQwD//+c+Ca0gkEnj11VfhcrlwyimnAADi8TiWLFmCK664ApIkQRAEdnwqlcLOnTuLvp67NlVVHffz+Xzs53vuuQefffYZzjnnHMyaNQuapoEQwj6P/fqbNm3CPvvsg5aWFtx333344IMPkEgkcPTRRxf8XBwcueAeCMdehbPOOgvLly/HG2+8kZchdcIJJ+CNN95g8f0XX3wR9fX1mDhxIk444QQ899xzAIBdu3axp/4DDjgAHo+HEcju3bsxb948rFmzpugali1bhvr6evztb3/D22+/jbfffhtvvfUWEokEli9fjlmzZmHlypVoa2sDADz77LO45557ir7e2NiIXbt2IRwOgxCC119/vei933vvPVx66aWYP38+gsEgVqxYAV3X4ff7ceihh+KVV15hn+Pf/u3fEIvF4PV6ceaZZ+KWW27BBRdc0JevnWOUgisQjr0Kzc3NOPDAAxEIBFBfX+9477jjjsNll12GSy+9FIZhoLGxEY899hhEUcRPf/pT3HzzzTj99NMxduxYHHzwwQAAl8uFhx9+GHfddReeeOIJaJqG73//+zjqqKOKhpaeeeYZXH755ZAkib1WW1uLiy++GE899RRefPFFLFiwgKX3hkIh/PznP0dzc3PR1y+44AKcc845CIVCOPnkk7F69eqC9/7e976Hu+++Gw888AAURcGRRx6Jbdu2AQAWLVqEO+64A7///e8hCALuuusuhEIhAGY4b+nSpZg/f34/vn2O0QaBEN7OnYNjNIMQgscf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9ilz3UC+CORyOU499VR27tw5KdfsH//xHznmmGPiz9XurfLlra2t437tysc2d+5cPvaxjwHQ2dnJfffdxyc+8Qn6+/s55JBDuPzyy3n00Ufp6enh9ttvn7BxVfvtJvo5LR9XxNtvv81zzz3HhRdeCDDh16vSHKFp2qTcZ0qAMHzhx8ngjTfeYPXq1VxxxRXsv//+3HbbbbS1tZFOp/nsZz/Lf/3Xf03oeI466ihuuOEGGhoaaGlp4ZxzzuE73/nOlLpuDzzwABdffDEA8+fPn/RrBtXvral0z7W3t3PRRRdx9tlns2TJEurq6vj+97/PAQccgGmarF69ekKvXbXfbqpcswcffJDzzz8f27YBJu16JeeI+fPnT8p9pgQIwxd+nGheeOEF/vIv/5K//du/5dOf/jSvv/56XBsM5E0w3s7Wcp5//nk2btxYMoa5c+dOmetWKBT49a9/HXe2nArXDKrfW+XLd+7cOSnX7o9//COrVq3i05/+NF/60pcA2L59e4m2NtHXrtpvN1We05///Oecdtpp8efJuF7lc8Rk3WdKgDB84ceJ5L333uNLX/oSN910E6effjogb8hvfvObdHd347ouDz744IRHYPX29nLDDTeQz+fp6+vj0Ucf5cYbb5wy1+31119n3333jYtuToVrBnDEEUfw1ltv8c477+D7Pk8++STLli1j7ty5OI7DCy+8AMDjjz8+4deur6+Pz33uc3zlK19h9erV8fJUKsWNN97Ili1bEEJw3333Tei1q/bbVbuWE0lnZye5XI758+fHyyb6elWaIybrPlNhvJQWfnRdl3POOYfDDz98Usbyb//2b+Tzea6//vp42apVq/j85z/Peeedh+d5nHLKKaxYsWJCx3XiiSfy4osvcuaZZxIEAeeffz4f/vCHp8x127JlC7NmzYo/H3zwwZN+zQAcx+H666/nr//6r8nn85xwwgl86lOfAuCmm27i6quvpq+vj0WLFsU29Yni4YcfZufOndx1113cddddAJx00kl85StfYe3atVx66aW4rsvRRx8dmwYngqF+u2rXcqLYunVryX0G0NLSMqHXq9ocMRn3mSqmqFAoFIqaUCYshUKhUNSEEiAKhUKhqAklQBQKhUJRE0qAKBQKhaImlABRKBQKRU2oMF6FYhR86EMf4qCDDkLXS9+9brvtNubNmzfmx9q4cSMtLS1jul+FYqxQAkShGCV33323mtQVCpQAUSjGjGeffZabbrqJOXPmsHnzZlKpFNdffz0HHHAAvb29XHvttbz22mtomsbxxx/PV7/6VUzT5MUXX+S6664jm81iWRZXXHEFH/3oRwH47ne/y4svvsju3bv53Oc+N6l9ahSKcpQAUShGyUUXXVRiwpo3bx633XYbIMt6f+1rX+OYY47h/vvv5/LLL+eRRx7huuuuo7m5mXXr1uG6Lpdeein//u//zsUXX8yXvvQlrrvuOj7+8Y+zadMmrrzySh5//HFAFha85ppreOWVVzj33HP5zGc+g2VZk3LeCkU5SoAoFKNkKBPWwQcfHJcAP/vss1m7di1dXV3893//N/fffz+apmHbNqtWreLuu+/mYx/7GLqu8/GPfxyAww47jHXr1sX7i0p4HHLIIRQKBfr6+pg2bdr4nqBCMUJUFJZCMYYYhlFxWXlZ7SAI8DwPwzAGldf+wx/+gOd5AHFV12gdVXlIMZVQAkShGENee+01XnvtNUD2jTjqqKNobGzkuOOO495770UIQaFQ4KGHHuLYY49l//33R9M0fvWrXwHw+9//nosuumjCO04qFLWgTFgKxSgp94EAfPWrXyWVSjFjxgy+/e1vs23bNlpaWrjhhhsAuPrqq7nuuutYuXIlruty/PHH88UvfhHbtvnud7/LN7/5TW644QYsy+K73/1u3KxIoZjKqGq8CsUY8eyzz/IP//APPPnkk5M9FIViQlAmLIVCoVDUhNJAFAqFQlETSgNRKBQKRU0oAaJQKBSKmlACRKFQKBQ1oQSIQqFQKGpCCRCFQqFQ1IQSIAqFQqGoif8f1VqszNfuqvkAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(history.history['acc'])\n",
    "plt.plot(history.history['val_acc'])\n",
    "plt.title('Model Accuracy')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.xlabel('Epoch')\n",
    "plt.legend(['train', 'test'], loc='upper left')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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06NHD6D92qup34ecnJyAfcc6VaEoLi7Gbrc3XE9MTKSoqCggWU455ZSGDzYnJ4cvvviCCRMmMHbsWJ566ikWLlzIjz/+yAcffNCuuaqqqjjttNN4+eWXeeONN3j33XfJz88PmnaD+oLvdDqZNm0apaWlAWmzefPmMWrUqIbrR/tu/Xq73W73e9v9OltKSgrjx48HoLy8nLfeeouzzjqL2tpa+vfvz7333svHH39MVVUVf//739st19E+u/b+d/rrXIfl5OSwbt06rrnmGoB2ba+mfh80TQvYd6xTFQrDMNC0n6fZVUo1uh4Iu3fvZvbs2dx333306tWLl19+mcTEREJDQ7n66qv57rvv2jXP8OHDee6554iMjCQuLo6LL76Yl156Kaja7d133+X6668HoFu3bgFvMzj6dyuYvnNFRUVce+21XHTRRYwZM4bw8HBee+01evfujdlsZvbs2e3adkf77IKlzd577z2uvPJKrFYrQEDa65e/D926dQvYd6xTFYqkpCRKSkoarpeUlDS5u9le1q9fz3XXXcfvfvc7Zs2axc6dO1myZEnD7Uopv5/0/LUff/yR1atXN8qQkpISNO3mdrv54YcfmDx5MkBQtBkc/bv16+2lpaUBabu9e/dy+eWXM2vWLG6//XYA8vPzG+19tXfbHe2zC5Z/p19//TXTp09vuN7e7fXr34dAfsc6VaEYN24cq1evpry8HIfDwdKlS5k4cWJAshQUFHD77bfzwgsvcN555wH1X7ynnnqKyspKPB4P7733Xrv3eKqurua5557D5XJRU1PDxx9/zPPPPx807bZz50569OhBWFgYEBxtBjB06FCys7PZt28fPp+Pzz//nIkTJ5KSkoLNZmP9+vUAfPrpp+3edjU1Ndxwww3ceeedzJ49u2F7SEgIzz//PLm5uSileOutt9q17Y722R2tLdtTeXk5TqeTbt26NWxrz/Zq6vchkN+xTtU9tkuXLtx9991cc801eDweLr74YoYMGRKQLK+//joul4tnnnmmYdvll1/OTTfdxBVXXIHX62XKlClkZGS0a64zzzyTTZs2ccEFF2AYBldeeSUjR44MmnbLzc0lKSmp4Xp6enrA2wzAZrPxzDPP8Nvf/haXy8WkSZM499xzAXjhhRd45JFHqKmpYeDAgQ3HvNvLBx98QGlpKfPnz2f+/PkATJ48mTvvvJMnnniCW2+9FY/Hw4gRIxoO6bWH5j67o7Vle8nLy2v0PQOIi4trt/Y62u9DoL5jssKdEEKIZnWqQ09CCCFaTgqFEEKIZkmhEEII0SwpFEIIIZolhUIIIUSzpFAIESBr164NSFdeIVpKCoUQQohmdaoBd0K0xLJly/jHP/6Bx+MhJCSE+++/nxUrVrBv3z4KCwspKSkhPT2defPmERERwe7du3niiSeoqKhA0zRmz57NBRdcANQPeps/fz66rhMbG8uzzz4LQF1dHXfffTdZWVm4XC6efPLJJieoEyKg2nTSciE6iOzs7P/f3h27JBdGcRz/Wjm6uTS0REuTk5hpgQTSEC41S0NB4CY01iAiQosg1Nhc0B9g0BAh1iJuTkIIFhjk4JLm9TRcEOKVSy35Dr/PeO7l4Z7pcO4D59jOzo69v7+bmbv/IhaLWbFYtM3NTXt7ezPHceqdS5gAAAG7SURBVCybzVqxWLTPz0/b2tqySqViZu4Oj42NDavX69ZsNi0SidjLy4uZmV1eXtrJyYk9Pj7a6uqqNRqNSTydTs8mYREP6ihEpqhWq3S7Xfb39ycxn89Hu91me3ubYDAIwN7eHoVCgd3dXQaDAclkEnDHxSSTSR4eHggEAsTjcRYXFwEmZz49PbG0tEQoFALckRY3Nzd/l6TID6lQiEwxHo+JRqOUSqVJ7PX1laurK4bD4bf35ubmcBznn9HOZsZoNGJ+fv7bs4+PDzqdDgB+v38S9/l8mCbqyH9Il9kiU0SjUarVKq1WC4D7+3tSqRSDwYC7uzv6/T7j8Zjr62sSiQTLy8ssLCxwe3sLuLsfKpUK6+vrRCIRarUa3W4XcPdpnJ2dzSw3kd9SRyEyxcrKCrlcjmw2O9k7cHFxQa1WIxgMcnh4SK/XIxwOc3R0hN/v5/z8nHw+T7lcxnEcMpkMa2trABwfH3NwcAC4G8gKhQLPz88zzFDk5zQ9VuQXyuUyvV6P09PTWX+KyJ/RrycREfGkjkJERDypoxAREU8qFCIi4kmFQkREPKlQiIiIJxUKERHxpEIhIiKevgDTXxwVAjYcvgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(history.history['loss'])\n",
    "plt.plot(history.history['val_loss'])\n",
    "plt.title('model loss')\n",
    "plt.ylabel('loss')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['train', 'test'], loc='upper left')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n"
     ]
    }
   ],
   "source": [
    "Y_pred = classifier.predict(X_test, batch_size=10, verbose=0)\n",
    "#Y_pred = (Y_pred>0.5)\n",
    "#Y_pred = [ 1 if y>=0.5 else 0 for y in Y_pred ]\n",
    "print(np.argmax(Y_pred[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "from sklearn.metrics import confusion_matrix\n",
    "import itertools\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.9385966  0.00757086]\n",
      "[0.5933341  0.25290278]\n",
      "[0.9488785  0.00905716]\n",
      "[0.3481377  0.46174636]\n",
      "[9.988253e-01 2.965331e-05]\n",
      "[0.19889441 0.641613  ]\n",
      "[0.6602358  0.19965893]\n",
      "[0.20610517 0.46647084]\n",
      "[0.9391173  0.00788463]\n",
      "[0.19075648 0.66586834]\n",
      "[0.3045964  0.54002714]\n",
      "[0.79016405 0.05969414]\n",
      "[0.97892237 0.00212458]\n",
      "[9.9948144e-01 8.7618828e-06]\n",
      "[0.72367024 0.10738161]\n",
      "[0.08388317 0.78591096]\n",
      "[0.9401173  0.01115745]\n",
      "[9.9988467e-01 1.3411045e-06]\n"
     ]
    }
   ],
   "source": [
    "for i in Y_pred:\n",
    "    print(i)   # data prediction for output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "Y_pred_rounded = classifier.predict_classes(X_test, batch_size=10, verbose=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n",
      "0\n",
      "0\n",
      "1\n",
      "0\n",
      "1\n",
      "0\n",
      "1\n",
      "0\n",
      "1\n",
      "1\n",
      "0\n",
      "0\n",
      "0\n",
      "0\n",
      "1\n",
      "0\n",
      "0\n"
     ]
    }
   ],
   "source": [
    "for i in Y_pred_rounded:\n",
    "    print(i)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[6, 1],\n",
       "       [6, 5]], dtype=int64)"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confusion_matrix(y_test, Y_pred_rounded) # Confusion matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x114156df4e0>"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "labels = ['True Neg','False Pos','False Neg','True Pos']\n",
    "labels = np.asarray(labels).reshape(2,2)\n",
    "sns.heatmap(confusion_matrix(y_test, Y_pred_rounded), annot=labels, fmt='', cmap='Blues')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [],
   "source": [
    "classifier.save('weights.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [],
   "source": [
    "def converter(value):\n",
    "    if value == 0:\n",
    "        return \"Negative\"\n",
    "    else:\n",
    "        return \"Positive\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Making prediction from the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Positive\n"
     ]
    }
   ],
   "source": [
    "ex_measures=np.array([0.07722,\n",
    "0.962,\n",
    "0.2792,\n",
    "0.9639,\n",
    "1.6119])\n",
    "ex_measures=ex_measures.reshape(1,-1)\n",
    "\n",
    "prediction = classifier.predict(ex_measures)\n",
    "print(converter(np.argmax(prediction)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# It can be said the neural network gives an accurate answear"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
