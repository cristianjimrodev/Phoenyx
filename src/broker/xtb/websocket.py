"""Low-level WebSocket connection to XTB xAPI."""
from __future__ import annotations

import asyncio
import json
import time

import websockets
from loguru import logger


class XTBWebSocket:
    """Manages a single WebSocket connection to xAPI."""

    def __init__(self, url: str):
        self._url = url
        self._ws = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected and self._ws is not None

    async def connect(self) -> None:
        try:
            self._ws = await websockets.connect(self._url, max_size=None)
            self._connected = True
            logger.info(f"WebSocket connected to {self._url}")
        except Exception as e:
            self._connected = False
            logger.error(f"WebSocket connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()
            self._connected = False
            logger.info("WebSocket disconnected")

    async def send(self, command: str, arguments: dict | None = None,
                   custom_tag: str = "") -> dict:
        if not self.connected:
            raise ConnectionError("WebSocket not connected")

        msg = {"command": command}
        if arguments:
            msg["arguments"] = arguments
        if custom_tag:
            msg["customTag"] = custom_tag

        raw = json.dumps(msg)
        logger.debug(f"TX: {command}")
        await self._ws.send(raw)

        response_raw = await asyncio.wait_for(self._ws.recv(), timeout=30)
        response = json.loads(response_raw)
        logger.debug(f"RX: {command} status={response.get('status')}")

        if not response.get("status", False):
            error_code = response.get("errorCode", "UNKNOWN")
            error_desc = response.get("errorDescr", "No description")
            raise RuntimeError(f"xAPI error [{error_code}]: {error_desc}")

        return response

    async def send_stream(self, command: str, arguments: dict | None = None) -> None:
        """Send a streaming command (no response expected)."""
        if not self.connected:
            raise ConnectionError("Streaming WebSocket not connected")

        msg = {"command": command}
        if arguments:
            # Streaming commands put args at top level, not nested
            msg.update(arguments)

        await self._ws.send(json.dumps(msg))
        logger.debug(f"Stream TX: {command}")

    async def receive_stream(self) -> dict:
        """Receive a streaming message."""
        if not self.connected:
            raise ConnectionError("Streaming WebSocket not connected")

        raw = await self._ws.recv()
        return json.loads(raw)


class XTBConnection:
    """Manages both command and streaming WebSocket connections to XTB."""

    def __init__(self, url: str, stream_url: str):
        self._cmd = XTBWebSocket(url)
        self._stream = XTBWebSocket(stream_url)
        self._stream_session_id: str | None = None
        self._heartbeat_task: asyncio.Task | None = None

    @property
    def command(self) -> XTBWebSocket:
        return self._cmd

    @property
    def stream(self) -> XTBWebSocket:
        return self._stream

    async def login(self, user_id: str, password: str) -> str:
        """Connect both sockets and authenticate. Returns stream session ID."""
        await self._cmd.connect()

        response = await self._cmd.send("login", {
            "userId": user_id,
            "password": password,
        })

        self._stream_session_id = response.get("streamSessionId")
        if not self._stream_session_id:
            raise RuntimeError("No streamSessionId received from login")

        logger.info(f"Authenticated as user {user_id}")

        await self._stream.connect()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        return self._stream_session_id

    async def logout(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        try:
            await self._cmd.send("logout")
        except Exception:
            pass

        await self._stream.disconnect()
        await self._cmd.disconnect()
        logger.info("Logged out from XTB")

    async def _heartbeat_loop(self, interval: int = 30) -> None:
        """Send periodic pings to keep the connection alive."""
        while True:
            try:
                await asyncio.sleep(interval)
                await self._cmd.send("ping")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
                break
