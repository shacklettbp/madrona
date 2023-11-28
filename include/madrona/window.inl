namespace madrona {

// Workaround for what seems to be a clang compiler bug:
// https://stackoverflow.com/questions/53408962/try-to-understand-compiler-error-message-default-member-initializer-required-be
WindowManager::Config WindowManager::defaultConfig() { return {}; }

}
