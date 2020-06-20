#include <glog/logging.h>
#include "sky_detector/imageSkyDetector.h"
#include "file_processor/file_system_processor.h"

#define BATCH_PROCESS

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::SetLogDestination(google::GLOG_INFO, "./log/image_quality_check_");
    google::SetStderrLogging(google::GLOG_INFO);

    //make log dir
    if(!file_processor::FileSystemProcessor::is_directory_exist("./log")){
        file_processor::FileSystemProcessor::create_directories("./log");
    }

    


}
