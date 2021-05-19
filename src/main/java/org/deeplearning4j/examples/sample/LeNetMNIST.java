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
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
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

/**
 * Created by agibsonccc on 9/16/15.
 */
public class LeNetMNIST {
    private static final Logger log = LoggerFactory.getLogger(LeNetMNIST.class);

    public static void main(String[] args) throws Exception {


        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains


        int nChannels = 8; // Number of input channels
        int outputNum = 2; // The number of possible outcomes
        int train_batch = 8;
        int test_batch = 8;// Test batch size, czyli rozmiar wejscia. Ustawiamy go tez dalej.
        int nEpochs = 100; // Number of training epochs - nie wiem jak to ustawic, gdzie to przekazac
        int seed = 123; // Seed generatora randomowego (?) , który miesza dane przed nauką i testem

        /*
            Create an iterator using the batch size for one iteration
         */
        log.info("Load data....");


        System.out.println();

        // try other loading than RecordReader... (from word2vecrawtextexample.java)
        String filePath = "C:\\Users\\kwozn\\Documents\\Projekty\\ml\\src\\main\\resources";//new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        //String filename = "SandPFormatAllRand.csv";


        //TODO: TO ZMIENIAMY NAZWE PLIKU
        String filename1 = "data_train.csv";
        String filename2 = "data_test.csv";
        boolean my_dataset = false; // test dataset
        if(filename1 == "data_train.csv") my_dataset = true;
        if(filename2 == "data_test.csv") my_dataset = true;
        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader recordReader1 = new CSVRecordReader();
        RecordReader recordReader2 = new CSVRecordReader();
        //TODO: TU ZMIENIAMY SCIEZKE
        recordReader1.initialize(new FileSplit(new File("D:\\PP\\mvn-project-template\\src\\main\\resources\\"+filename1)));
        //recordReader.initialize(new FileSplit(new ClassPathResource("my_dataset.txt").getFile())); // FileNotFound exception
        recordReader2.initialize(new FileSplit(new File("D:\\PP\\mvn-project-template\\src\\main\\resources\\"+filename2)));


        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 0;
        int numClasses = 0;

        if(my_dataset){
            labelIndex = 8;     //5 values in each row of the my_dataset.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
            numClasses = 2;     //3 classes (types of my_dataset flowers) in the my_dataset data set. Classes have integer values 0, 1 or 2
            //TODO: TU USTAWIC ROZMIAR WEJSCIA
        }

//        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator.Builder()
//                .classification(labelIndex, numClasses)
//                .build()
        DataSetIterator train_iterator = new RecordReaderDataSetIterator(recordReader1,train_batch,labelIndex,numClasses);
        DataSetIterator test_iterator = new RecordReaderDataSetIterator(recordReader2,test_batch,labelIndex,numClasses);
        int numInputs = labelIndex;


        DataSet trainingData = train_iterator.next();
        DataSet testData = test_iterator.next();

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

        /*
            Construct the neural network
         */
        log.info("Build model....");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
               // .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.2,0.5))
                .list()
                //Ponizej mamy warstwe wejsciowa
                .layer(0,new DenseLayer.Builder()
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .nOut(5)
                        .weightInit(WeightInit.UNIFORM)
                        .activation(Activation.SIGMOID)
                        .build())
                //Warstwa ukryta
                .layer(1,new DenseLayer.Builder()
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        //Pozwoliłem sobie ustawić 5 neuronow w ukrytej
                        .nIn(5)
                        .nOut(5)
                        //Uniform czyli rozklad gaussa
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.RELU)
                        .build())
                //Warstwa wyjsciowa
                .layer(2,new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        // No wejść chyba tyle musi być co w poprzedniej warstwie
                        .nIn(5)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.UNIFORM)
                        .build())
                .build();

        /*
        Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
        (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
            and the dense layer
        (b) Does some additional configuration validation
        (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
            layer based on the size of the previous layer (but it won't override values manually set by the user)

        InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
        For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
        MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
        row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
        */

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Train model...");

        model.setListeners(new MyEvaluativeListener(train_iterator,1, InvocationType.EPOCH_END)); //Print score every 10 iterations and evaluate on test set every epoch
        model.setListeners(new StatsListener(statsStorage));
        System.out.println(train_iterator.toString());

        model.fit(train_iterator,100);

        String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "test_model.zip");

        log.info("Saving model to tmp folder: "+path);

        Evaluation evaluation = model.evaluate(test_iterator);
        System.out.println(evaluation.stats());

        model.save(new File(path), true);

        log.info("****************Example finished********************");


        new Scanner(System.in).nextInt();
    }
}
