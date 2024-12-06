# SDKs

### Description 
This is a repository that exclusively uses Git LFS. The purpose of this SDK is to contain the camera SDKs used by the CellVista software. This repository is used in the CellVista GitHub repository as a submodule. This is because in the event that this repository hits the Git LFS limit it can be deleted and started fresh without significantly affecting the CellVista repository (Similar to what happened with GitLab and its 10GB size limit).

When this repository is modified, the CellVista repository submodule should also be modified to point to the correct repository/head.