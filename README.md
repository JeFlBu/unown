# unown

# SciFact
You need Multivers in order to run inference on the generated code, as well as generating claims

You can perform

git clone https://github.com/dwadden/multivers.git

Once downloaded, you should run

bash script/get_data_train.sh

In order to get the training/test data and corpus from SciFact necessary to run our generation

You should replace base_path_multivers in scifact/claimgen.ipynb by the actual path of your multivers

Finally, replace base_path value by your working directory



# Feverous
You need the feverous code to run our experiments https://github.com/Raldir/FEVEROUS.git

You will also need the feverous feverouswikiv1.db dataset from https://fever.ai/download/feverous/feverous-wiki-pages-db.zip

claimgen.ipynb allows you to construct new examples

entityreplacement.ipynb permits to replace the negative of a generated/original set of claims

runPipeline.ipynb serves to run the pipeline of Feverous in a way that uses only the predictor part, and not the retriever. It will train and test it. The test is run on original dataset. It gives an output file of predicted labels and another one containing various metrics including th accuracy on the fact-checking pipeline

