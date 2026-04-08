# Лабораторная работа №1

## 1. Переписать существующий пайплайн на основе среды SourceCraft

У меня есть выполненная работа, где нужно было разместить hugo-site на GitHub:  
[https://github.com/Kidallow/hugosite.github.io](https://github.com/Kidallow/hugosite.github.io)

И этот сайт я перенес на SourceCraft:  
[https://sourcecraft.dev/kidallowv/demo1803](https://sourcecraft.dev/kidallowv/demo1803)

## 2. Дополнить сценарии проверками синтаксиса Markdown (в виде отдельных шагов)

Дальше с помощью нейросети я смог сделать `markdownlint` в файле `ci.yaml`, который проверяет контроль качества еще до сборки проекта: если что-то не так — проект даже не начнет собираться.
