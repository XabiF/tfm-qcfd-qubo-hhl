@echo off
setlocal enabledelayedexpansion

for /D %%F in (frames*) do (
    if exist "%%F" (
        echo Deleting frames dir %%F
        rmdir /s /q "%%F"
    )
)

for /D %%F in (*steps) do (
    if exist "%%F" (
        echo Deleting steps dir %%F
        rmdir /s /q "%%F"
    )
)

for %%G in (*.gif) do (
    if exist "%%G" (
        echo Deleting GIF %%G
        del /q "%%G"
    )
)
