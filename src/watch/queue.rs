use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Debug, Default)]
pub struct CoalescingQueue { paths: BTreeMap<PathBuf, Instant>, dirty: bool }
#[derive(Debug, PartialEq, Eq)]
pub enum QueueInsert { Added, Coalesced, Overflow }
impl CoalescingQueue {
    pub fn insert(&mut self, path: PathBuf, now: Instant, capacity: usize) -> QueueInsert {
        let is_full = self.paths.len() >= capacity;
        match self.paths.entry(path) {
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                entry.insert(now);
                QueueInsert::Coalesced
            }
            std::collections::btree_map::Entry::Vacant(_) if is_full => {
                self.dirty = true;
                QueueInsert::Overflow
            }
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(now);
                QueueInsert::Added
            }
        }
    }
    pub fn drain_due(&mut self, now: Instant, debounce: Duration) -> Vec<PathBuf> {
        let due: Vec<_> = self.paths.iter().filter(|(_, seen)| now.saturating_duration_since(**seen) >= debounce).map(|(path, _)| path.clone()).collect();
        for path in &due { self.paths.remove(path); }
        due
    }
    pub fn len(&self) -> usize { self.paths.len() }
    pub fn is_dirty(&self) -> bool { self.dirty }
    pub fn clear(&mut self) {
        self.paths.clear();
        self.dirty = false;
    }
}
