use std::fs::File;
use std::path::Path;
use fs2::FileExt;
use crate::state::{Result, StateError};

#[derive(Debug)]
pub struct OwnerLock {
    _file: File,
}

impl OwnerLock {
    pub fn acquire(data_dir: &Path) -> Result<Self> {
        let lock_path = data_dir.join("owner.lock");
        
        // Ensure data directory exists
        std::fs::create_dir_all(data_dir)?;
        
        // Open or create the lock file
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;
            
        // Try to acquire an exclusive lock without blocking
        match file.try_lock_exclusive() {
            Ok(_) => Ok(Self { _file: file }),
            Err(e) => {
                // If it's a WouldBlock error, another process has the lock
                if e.kind() == std::io::ErrorKind::WouldBlock {
                    Err(StateError::Busy)
                } else {
                    Err(StateError::Io(e))
                }
            }
        }
    }
}

impl Drop for OwnerLock {
    fn drop(&mut self) {
        let _ = self._file.unlock();
    }
}

