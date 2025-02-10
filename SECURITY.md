# Using PaddlePaddle Securely

This document describes model security and code security in PaddlePaddle. It also provides guidelines on how to report vulnerabilities in PaddlePaddle.

## PaddlePaddle Model Security

PaddlePaddle attaches great importance to security and privacy of model. This includes how to prevent the model from outputting wrong decision results under the interference when it is used in security-related and safety-critical scenarios, and how to avoid leaking data and privacy information from the model itself, the model gradient or the model inference results.



[PaddleSleeve](https://github.com/PaddlePaddle/PaddleSleeve) provides a series of security and privacy tools, which can help model developers and users systematically evaluate and improve the model security and privacy in both development and deployment stages.



These tools include adversarial example evaluation test, pseudo-natural environment robustness evaluation test, model reversing evaluation test, member inference evaluation test, sample denoising, adversarial training, privacy enhancement optimizer, etc.

### Running untrusted models

Always load and execute untrusted models inside a sandbox and be sure to know the security impacts.
There are several ways in which a model could become untrusted. PaddlePaddle has enough features to impact on the system. (e.g. `paddle.load` uses [pickle](https://docs.python.org/3/library/pickle.html) implicitly, which may cause malformed models to achieve arbitrary code execution). So we recommend when using the untrusted models, you need to carefully audit it and run PaddlePaddle inside a sandbox.

## PaddlePaddle Code Security

PaddlePaddle always take code security seriously. However, due to the complexity of the framework and its dependence on other thirdparty open source libraries, there may still be some security issues undetected. Therefore, we hope that more security researchers and PaddlePaddle developers can participate in the code security program. We encourage responsible disclosure of security issues, as well as contributing code to improve our vulnerability finding tools to make PaddlePaddle safer.

### Code security tools

PaddlePaddle security team attaches great importance to the security of the framework. In order to find and fix security issues as soon as possible, we are continuously conducting code security audit and developing automatic vulnerability discovery tools. We have already open sourced some of them to the community, hoping this could encourage people to contribute and improve the safety and robustness of PaddlePaddle. [This tool](https://github.com/PaddlePaddle/PaddleSleeve/tree/main/CodeSecurity) includes two parts. The dynamic part includes some op fuzzer samples. And the static part includes some CodeQL samples. Both of them are aim to find vulnerabilities in PaddlePaddle framework codebase. By referring the samples, security researchers can write their own fuzzers or QLs to test more PaddlePaddle modules, and find more code security issues.

### Reporting vulnerabilities

We encourage responsible disclosure of security issues to PaddlePaddle and please email reports about any security issues you find to paddle-security@baidu.com.



After the security team receives your email, they will communicate with you in time. The security team will work to keep you informed of an issue fix.



In order to reproduce and identify the issue, please include the following information along with your email:

- The details of the vulnerability including how to reproduce it. Try to attach a PoC.
- The attack scenario and what an attacker might be able to achieve with this issue.
- Whether this vulnerability has been made public. If it is, please attach details.
- Your name and affiliation.

We will indicate the bug fix in the release of PaddlePaddle, and publish the vulnerability detail and the reporter in the security advisories (Your name will not be published if you choose to remain anonymous).

### What is a vulnerability?

In the process of computation graphs in PaddlePaddle, models can perform arbitrary computations , including reading and writing files, communicating with the network, etc. It may cause memory exhaustion, deadlock, etc., which will lead to unexpected behavior of PaddlePaddle. We consider these behavior to be security vulnerabilities only if they are out of the intention of the operation involved.



Some unexpected parameters and behaviors have been checked in PaddlePaddle by throwing exceptions in Python or return error states in C++. In these cases, denial of service is still possible, but the exit of the PaddlePaddle is clean. Since the error handling of PaddlePaddle is expected and correct, these cases are not security vulnerabilities.



If malicious input can trigger memory corruption or non-clean exit, such bug is considered a security problem.



[security advisories](./security/README.md)
