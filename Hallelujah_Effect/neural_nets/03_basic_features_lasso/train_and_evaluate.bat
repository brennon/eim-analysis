@ECHO OFF

SET ARG=%1

SET MODEL_NAME=basic_features_local

SET OLDPYTHONPATH=%PYTHONPATH%
SET PYTHONPATH=%PYTHONPATH%;%cd%\trainer

SET TRAIN_N=303

ECHO ********************************************************************************
ECHO * Training %MODEL_NAME% locally
ECHO ********************************************************************************

SET MODEL_DIR=%cd%\model_trained
ECHO Removing %MODEL_DIR%
rmdir /S /Q %MODEL_DIR%

python ^
-m trainer.task ^
--train_data_paths=%cd%\..\sample\train* ^
--eval_data_paths=%cd%\..\sample\eval* ^
--output_dir=%MODEL_DIR% ^
--num_train_examples=303 ^
--train_batch_size=303 ^
--train_steps=1000 ^
--job-dir=C:\Windows\Temp
REM --optimize=false

REM ECHO ********************************************************************************
REM ECHO * Training %MODEL_NAME% locally (with ml-engine)
REM ECHO ********************************************************************************
REM
REM SET MODEL_DIR=%cd%\model_trained
REM ECHO Removing %MODEL_DIR%
REM rmdir /S /Q %MODEL_DIR%
REM
REM gcloud ml-engine local train ^
REM --module-name=trainer.task ^
REM --package-path=%cd%\trainer\trainer ^
REM --job-dir=%MODEL_DIR% ^
REM -- ^
REM --train_data_paths=%cd%\sample\train* ^
REM --eval_data_paths=%cd%\sample\eval* ^
REM --output_dir=%MODEL_DIR% ^
REM --train_steps=50000 ^
REM --train_batch_size=%TRAIN_N% ^
REM --eval_steps=1 ^
REM --checkpoint_secs=10 ^
REM --hidden_units="4" ^
REM --learning_rate=0.0618 ^
REM --dropout=0.749 ^
REM --activation_function="elu"

REM ECHO ********************************************************************************
REM ECHO * Training %MODEL_NAME% on GCP (with hyperparameter tuning)
REM ECHO ********************************************************************************
REM
REM for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
REM SET "YY=%dt:~2,2%" & SET "YYYY=%dt:~0,4%" & SET "MM=%dt:~4,2%" & SET "DD=%dt:~6,2%"
REM SET "HH=%dt:~8,2%" & SET "Min=%dt:~10,2%" & SET "Sec=%dt:~12,2%"
REM
REM SET "fullstamp=%YYYY%%MM%%DD%%HH%%Min%%Sec%"
REM
REM SET JOBNAME=%MODEL_NAME%_%fullstamp%
REM SET OUTPUT_DIR=gs://eim-muse/analysis/hallelujah-effect/models/%JOBNAME%
REM SET REGION=us-east1
REM
REM ECHO Output directory: %OUTPUT_DIR%
REM ECHO Region: %REGION%
REM ECHO Job name: %JOBNAME%
REM REM ECHO Removing %OUTPUT_DIR%
REM
REM REM CMD /C gsutil -m rm -rf %OUTPUT_DIR%
REM
REM gcloud ml-engine jobs submit training %JOBNAME% ^
REM --region=%REGION% ^
REM --package-path=%cd%/trainer/trainer ^
REM --module-name=trainer.task ^
REM --job-dir=%OUTPUT_DIR% ^
REM --scale-tier=STANDARD_1 ^
REM --runtime-version=1.10 ^
REM --config=hyperparam.yaml ^
REM -- ^
REM --train_data_paths=gs://eim-muse/analysis/hallelujah-effect/samples/%MODEL_NAME%/train* ^
REM --eval_data_paths=gs://eim-muse/analysis/hallelujah-effect/samples/%MODEL_NAME%/eval* ^
REM --output_dir=%OUTPUT_DIR% ^
REM --train_steps=2500 ^
REM --train_batch_size=%TRAIN_N% ^
REM --eval_steps=1

REM ECHO ********************************************************************************
REM ECHO * Training %MODEL_NAME% on GCP (without hyperparameter tuning)
REM ECHO ********************************************************************************
REM
REM for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
REM SET "YY=%dt:~2,2%" & SET "YYYY=%dt:~0,4%" & SET "MM=%dt:~4,2%" & SET "DD=%dt:~6,2%"
REM SET "HH=%dt:~8,2%" & SET "Min=%dt:~10,2%" & SET "Sec=%dt:~12,2%"
REM
REM SET "fullstamp=%YYYY%%MM%%DD%%HH%%Min%%Sec%"
REM
REM SET JOBNAME=%MODEL_NAME%_%fullstamp%
REM SET OUTPUT_DIR=gs://eim-muse/analysis/hallelujah-effect/models/%JOBNAME%
REM SET REGION=us-east1
REM
REM ECHO Output directory: %OUTPUT_DIR%
REM ECHO Region: %REGION%
REM ECHO Job name: %JOBNAME%
REM REM ECHO Removing %OUTPUT_DIR%
REM
REM REM CMD /C gsutil -m rm -rf %OUTPUT_DIR%
REM
REM gcloud ml-engine jobs submit training %JOBNAME% ^
REM --region=%REGION% ^
REM --package-path=%cd%/trainer/trainer ^
REM --module-name=trainer.task ^
REM --job-dir=%OUTPUT_DIR% ^
REM --scale-tier=STANDARD_1 ^
REM --runtime-version=1.10 ^
REM -- ^
REM --train_data_paths=gs://eim-muse/analysis/hallelujah-effect/samples/%MODEL_NAME%/train* ^
REM --eval_data_paths=gs://eim-muse/analysis/hallelujah-effect/samples/%MODEL_NAME%/eval* ^
REM --output_dir=%OUTPUT_DIR% ^
REM --train_steps=5000 ^
REM --train_batch_size=%TRAIN_N% ^
REM --eval_steps=1 ^
REM --hidden_units="4" ^
REM --learning_rate=0.0618 ^
REM --dropout=0.749 ^
REM --activation_function="elu"

SET PYTHONPATH=%OLDPYTHONPATH%
