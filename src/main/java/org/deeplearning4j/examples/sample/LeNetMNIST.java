/* *****************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.deeplearning4j.examples.sample;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;

import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Scanner;


public class LeNetMNIST {
    private static final Logger log = LoggerFactory.getLogger(LeNetMNIST.class);

    public static void main(String[] args) throws Exception {


        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();       // new FileStatsStorage(File)

        uiServer.attach(statsStorage);

        int nChannels = 4;
        int outputNum = 2;
        int train_batch = 2;
        int test_batch = 2;
<<<<<<< HEAD
        int nEpochs = 30;
=======
        int nEpochs = 35;
>>>>>>> 8e89fc6132318113f5cc6d33a4fd1a0bf16c85b3
        int seed = 123;

        log.info("Load data....");


<<<<<<< HEAD
        String filename1 = "BCE.csv";
        String filename2 = "BCE_T.csv";
=======
        String filename1 = "TF.csv";
        String filename2 = "TF_T.csv";
>>>>>>> 8e89fc6132318113f5cc6d33a4fd1a0bf16c85b3
        boolean my_dataset = false; // test dataset

        RecordReader recordReader1 = new CSVRecordReader();
        RecordReader recordReader2 = new CSVRecordReader();
        recordReader1.initialize(new FileSplit(new File("C:\\Users\\kwozn\\Documents\\Projekty\\ml\\" + filename1)));
        recordReader2.initialize(new FileSplit(new File("C:\\Users\\kwozn\\Documents\\Projekty\\ml\\" + filename2)));


        int labelIndex = 4;
        int numClasses = 2;


        DataSetIterator train_iterator = new RecordReaderDataSetIterator(recordReader1, train_batch, labelIndex, numClasses);
        DataSetIterator test_iterator = new RecordReaderDataSetIterator(recordReader2, test_batch, labelIndex, numClasses);
        int numInputs = labelIndex;


        DataSet trainingData = train_iterator.next();
        DataSet testData = test_iterator.next();


        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);
        normalizer.transform(trainingData);
        normalizer.transform(testData);

        log.info("Build model....");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(0.5))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(nChannels)
                        .nOut(2)
                        .weightInit(WeightInit.UNIFORM)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(2)
                        .nOut(3)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(3)
                        .nOut(2)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.UNIFORM)
                        .build())

                .build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Train model...");

        model.setListeners(new StatsListener(statsStorage), new EvaluativeListener(train_iterator, 1, InvocationType.EPOCH_END));
        System.out.println(train_iterator.toString());

        model.fit(train_iterator, nEpochs);

        String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "test_model.zip");

        log.info("Saving model to tmp folder: " + path);

        Evaluation evaluation = model.evaluate(test_iterator);
        System.out.println(evaluation.stats());

        model.save(new File(path), true);

        log.info("****************Example finished********************");


        new Scanner(System.in).nextInt();
    }
}
