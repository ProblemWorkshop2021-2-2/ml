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
import org.nd4j.linalg.api.ndarray.INDArray;
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

    //tutaj fragment z regresja liniowa
        //Aby obliczyc wartosc dla danego dnia, trzeba go przekazac jako argument
        Double dayValue = LinearRegression.predictForValue(1);
        System.out.println(dayValue);

        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();       // new FileStatsStorage(File)

        uiServer.attach(statsStorage);

        int nChannels = 22;
        int outputNum = 2;
        int train_batch = 5;
        int test_batch = 1;
        int nEpochs = 15;
        int seed = 1;


        log.info("Load data....");



        String filename1 = "TFS.csv";
        String filename2 = "TFS_T.csv";
        String filename3 = "tensorflow.csv";

        boolean my_dataset = false; // test dataset

        RecordReader recordReader1 = new CSVRecordReader();
        RecordReader recordReader2 = new CSVRecordReader();
        RecordReader recordReader3 = new CSVRecordReader();
        recordReader1.initialize(new FileSplit(new File("D:\\PP\\mvn-project-template\\newData\\" + filename1)));
        recordReader2.initialize(new FileSplit(new File("D:\\PP\\mvn-project-template\\newData\\" + filename2)));
        recordReader3.initialize(new FileSplit(new File("D:\\PP\\mvn-project-template\\newData\\" + filename3)));



        int labelIndex = 22;//21;
        int numClasses = 2;


        DataSetIterator train_iterator = new RecordReaderDataSetIterator(recordReader1, train_batch, labelIndex, numClasses);
        DataSetIterator test_iterator = new RecordReaderDataSetIterator(recordReader2, test_batch, labelIndex, numClasses);
        DataSetIterator classify_iterator = new RecordReaderDataSetIterator(recordReader3, test_batch, labelIndex, numClasses);

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
                .updater(new Adam(0.1))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(nChannels)
                        .nOut(5)
                        .weightInit(WeightInit.UNIFORM)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(5)
                        .nOut(7)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(7)
                        .nOut(outputNum)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.UNIFORM)
                        .build())

                .build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Train model...");

        model.setListeners(new StatsListener(statsStorage));//, new EvaluativeListener(train_iterator, 1, InvocationType.EPOCH_END));
        System.out.println(train_iterator.toString());

        model.fit(train_iterator, nEpochs);

        String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "test_model.zip");

        log.info("Saving model to tmp folder: " + path);

        Evaluation evaluation = model.evaluate(test_iterator);
        System.out.println(evaluation.stats());

        model.save(new File(path), true);

        log.info("****************Example finished********************");

        INDArray output = model.output(classify_iterator);

        log.info(output.toString());
        new Scanner(System.in).nextInt();
    }
}
