/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_baidu_paddle_inference_Predictor */

#ifndef _Included_com_baidu_paddle_inference_Predictor
#define _Included_com_baidu_paddle_inference_Predictor
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_baidu_paddle_inference_Predictor
 * Method:    predictorTryShrinkMemory
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Predictor_predictorTryShrinkMemory
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_baidu_paddle_inference_Predictor
 * Method:    predictorClearIntermediateTensor
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Predictor_predictorClearIntermediateTensor
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_baidu_paddle_inference_Predictor
 * Method:    createPredictor
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_baidu_paddle_inference_Predictor_createPredictor
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_baidu_paddle_inference_Predictor
 * Method:    getInputNum
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_baidu_paddle_inference_Predictor_getInputNum
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_baidu_paddle_inference_Predictor
 * Method:    getOutputNum
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_baidu_paddle_inference_Predictor_getOutputNum
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_baidu_paddle_inference_Predictor
 * Method:    getInputNameByIndex
 * Signature: (JJ)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_baidu_paddle_inference_Predictor_getInputNameByIndex
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     com_baidu_paddle_inference_Predictor
 * Method:    getOutputNameByIndex
 * Signature: (JJ)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_baidu_paddle_inference_Predictor_getOutputNameByIndex
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     com_baidu_paddle_inference_Predictor
 * Method:    getInputHandleByName
 * Signature: (JLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_baidu_paddle_inference_Predictor_getInputHandleByName
  (JNIEnv *, jobject, jlong, jstring);

/*
 * Class:     com_baidu_paddle_inference_Predictor
 * Method:    getOutputHandleByName
 * Signature: (JLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_baidu_paddle_inference_Predictor_getOutputHandleByName
  (JNIEnv *, jobject, jlong, jstring);

/*
 * Class:     com_baidu_paddle_inference_Predictor
 * Method:    runPD
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_inference_Predictor_runPD
  (JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif
#endif
