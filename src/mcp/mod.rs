mod server;

pub use server::WorkspaceIndexServer;

use crate::config::{Config, Paths};

pub async fn serve_stdio(paths: Paths, config: Config) -> Result<(), String> {
    WorkspaceIndexServer::serve_stdio(paths, config).await
}
