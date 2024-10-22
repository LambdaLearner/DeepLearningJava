
package Training_Class.djl.trainModel.training;

import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.EasyTrain;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import ai.djl.nn.core.Linear;
import ai.djl.engine.Engine;
import ai.djl.util.JsonUtils;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;
import java.util.Map;
import ai.djl.nn.convolutional.Convolution;
import ai.djl.nn.*;

import ai.djl.basicdataset.cv.classification.Cifar10;



public final class TrainCifar {
    
    protected static long limit = Long.MAX_VALUE;
    protected  static int epochs = 5;
    protected boolean isSymbolic;
    protected boolean preTrained;
    protected static String opDir ;
    protected String modelDir = null;
    protected static int  maxGpus = Engine.getInstance().getGpuCount();
    protected static int batchSize = maxGpus > 0 ? 32 * maxGpus : 32;

     
     private TrainCifar() {}

    public static void main(String[] args) throws IOException, TranslateException {
        
        try (Model model = Model.newInstance("cifar")) {
            model.setBlock(getModel());

            // get training and validation dataset
            RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN);//,arguments
            RandomAccessDataset validateSet = getDataset(Dataset.Usage.TEST);
        
            // get training configuration
            DefaultTrainingConfig config = getTrainingConfig();

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                Shape inputShape = new Shape(1,3, Cifar10.IMAGE_HEIGHT , Cifar10.IMAGE_WIDTH);
                trainer.initialize(inputShape);


                EasyTrain.fit(trainer, epochs, trainingSet, validateSet);

                trainer.getTrainingResult();
            } 
        }
    }
    public static Block getModel(){
         return new SequentialBlock()
            .add(Conv2d.builder()
                            .setKernelShape(new Shape(3, 3))
                            .setFilters(32)
                            .optPadding(new Shape(0,0))
                            .optBias(false)
                            .build())                
            .add(Activation::relu)
            .add(Conv2d.builder()  
                            .setFilters(32)
                            .setKernelShape(new Shape(3, 3))
                            .optPadding(new Shape(0,0))
                            .optBias(false)
                            .build())
            .add(Activation::relu)
            .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(2, 2)))
            .add(Conv2d.builder()
                            .setFilters(64)
                            .setKernelShape(new Shape(3, 3))
                            .optPadding(new Shape(0,0))
                            .optBias(false)
                            .build())
            .add(Activation::relu)
            .add(Conv2d.builder()
                            .setFilters(64)
                            .setKernelShape(new Shape(3, 3))
                            .optPadding(new Shape(0,0))
                            .optBias(false)
                            .build())
            .add(Activation::relu)
            .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(2, 2)))
            .add(Conv2d.builder()
                            .setFilters(128)
                            .setKernelShape(new Shape(4, 4))
                            .optPadding(new Shape(0,0))
                            .optBias(false)
                            .build())
            .add(Activation::relu)
            .add(Conv2d.builder()
                            .setFilters(128)
                            .setKernelShape(new Shape(4, 4))
                            .optPadding(new Shape(0,0))
                            .optBias(false)
                            .build())                
            .add(Activation::relu)
            .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(2, 2)))
            .add(Blocks.batchFlattenBlock())
            .add(Linear.builder().setUnits(128).build())
            .add(Activation::relu)
            .add(Linear.builder().setUnits(10).build());
        }

   
    private static DefaultTrainingConfig getTrainingConfig() {
        String opDir = "output/training";
        SaveModelTrainingListener listener = new SaveModelTrainingListener(opDir);
        listener.setSaveModelCallback(
            trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float accuracy = result.getValidateEvaluation("Accuracy");
                    model.setProperty("Accuracy", String.format("%.5f", accuracy));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                }
                );
        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optDevices(Engine.getInstance().getDevices(maxGpus))
                .addTrainingListeners(TrainingListener.Defaults.logging(opDir));
    }

    private static RandomAccessDataset getDataset(Dataset.Usage usage)
            throws IOException {
        Cifar10 cifar =
                Cifar10.builder()
                        .optUsage(usage)
                        .setSampling(batchSize, true)
                        .optLimit(limit)
                        .build();
        cifar.prepare(new ProgressBar());
        return cifar;
    }
}

    