@ECHO OFF

SET MODEL_NAME=basic_features_local

ECHO ********************************************************************************
ECHO * Training %MODEL_NAME% locally
ECHO ********************************************************************************

SET PYTHONPATH=%PYTHONPATH%;%cd%\trainer

SET TRAIN_N=303

SET MODEL_DIR=%cd%\model_trained
ECHO Removing %MODEL_DIR%
rmdir /S /Q %MODEL_DIR%

python ^
-m trainer.task ^
--train_data_paths=%cd%\sample\train* ^
--eval_data_paths=%cd%\sample\eval* ^
--output_dir=%MODEL_DIR% ^
--train_steps=100 ^
--job-dir=C:\Windows\Temp

REM ECHO ********************************************************************************
REM ECHO * Training %MODEL_NAME% locally (with ml-engine)
REM ECHO ********************************************************************************

REM SET MODEL_DIR=%cd%\model_trained
REM ECHO Removing %MODEL_DIR%
REM rmdir /S /Q %MODEL_DIR%

REM gcloud ml-engine local train ^
REM --module-name=trainer.task ^
REM --package-path=%cd%\trainer\trainer ^
REM --job-dir=%MODEL_DIR% ^
REM -- ^
REM --train_data_paths=%cd%\sample\train* ^
REM --eval_data_paths=%cd%\sample\eval* ^
REM --output_dir=%MODEL_DIR% ^
REM --train_steps=100 ^
REM --train_batch_size=%TRAIN_N% ^
REM --eval_steps=1

ECHO ********************************************************************************
ECHO * Training %MODEL_NAME% on GCP
ECHO ********************************************************************************

for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
SET "YY=%dt:~2,2%" & SET "YYYY=%dt:~0,4%" & SET "MM=%dt:~4,2%" & SET "DD=%dt:~6,2%"
SET "HH=%dt:~8,2%" & SET "Min=%dt:~10,2%" & SET "Sec=%dt:~12,2%"

SET "fullstamp=%YYYY%%MM%%DD%%HH%%Min%%Sec%"

SET JOBNAME=%MODEL_NAME%_%fullstamp%
SET OUTPUT_DIR=gs://eim-muse/analysis/hallelujah-effect/models/%JOBNAME%
SET REGION=us-east1

ECHO Output directory: %OUTPUT_DIR%
ECHO Region: %REGION%
ECHO Job name: %JOBNAME%
REM ECHO Removing %OUTPUT_DIR%

REM CMD /C gsutil -m rm -rf %OUTPUT_DIR%

gcloud ml-engine jobs submit training %JOBNAME% ^
--region=%REGION% ^
--package-path=%cd%/trainer/trainer ^
--module-name=trainer.task ^
--job-dir=%OUTPUT_DIR% ^
--scale-tier=STANDARD_1 ^
--runtime-version=1.10 ^
--config=hyperparam.yaml ^
-- ^
--train_data_paths=gs://eim-muse/analysis/hallelujah-effect/samples/%MODEL_NAME%/train* ^
--eval_data_paths=gs://eim-muse/analysis/hallelujah-effect/samples/%MODEL_NAME%/eval* ^
--output_dir=%OUTPUT_DIR% ^
--train_steps=10 ^
--train_batch_size=%TRAIN_N% ^
--eval_steps=1
